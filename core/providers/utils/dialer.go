package utils

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/maximhq/bifrost/core/network"
	"github.com/valyala/fasthttp"
)

// ConfigureDialer configures the client's connection behavior:
//  1. Sets up the stale-connection retry policy (see network.StaleConnectionRetryIfErr).
//  2. Wraps the Dial function to enable TCP keepalive on all connections,
//     proactively detecting dead connections before fasthttp tries to reuse them.
//
// Must be called AFTER ConfigureProxy (which may set client.Dial to a proxy
// dialer), so the keepalive wrapper composes on top of the proxy connection.
//
// Keepalive parameters:
//   - Idle 10s: first probe after 10s of inactivity (well under the 30s MaxIdleConnDuration)
//   - Interval 5s: subsequent probes every 5s
//   - Count 3: close after 3 failed probes
//
// Dead connections are detected within ~25s (10 + 5*3), before the 30s
// MaxIdleConnDuration expires and the connection is reused.
func ConfigureDialer(client *fasthttp.Client, allowPrivateNetwork bool) *fasthttp.Client {
	// A factory client forwards each request to an inner client that already carries
	// the factory's proxy, dialer and retry policy; it is left as it is.
	if network.IsLiveFasthttpClient(client) {
		return client
	}

	// Configure stale-connection retry policy
	client.RetryIfErr = network.StaleConnectionRetryIfErr

	// Every Bifrost client goes through the context-aware transport: it applies
	// the client's timeouts to the header phase only on streamed responses and
	// closes the socket when the request context ends (see roundtripper.go).
	client.Transport = NewContextTransport()

	existingDial := client.Dial
	existingDialTimeout := client.DialTimeout

	keepAliveCfg := net.KeepAliveConfig{
		Enable:   true,
		Idle:     10 * time.Second,
		Interval: 5 * time.Second,
		Count:    3,
	}

	client.Dial = func(addr string) (net.Conn, error) {
		var conn net.Conn
		var err error

		viaExistingDial := existingDial != nil
		bypassed := false
		if viaExistingDial {
			// Proxy or custom dial function is set — use it, then enable keepalive
			conn, err = existingDial(addr)
			// A no_proxy match on the proxy dialer: connect directly instead,
			// through the same checked path as a client with no proxy at all.
			if errors.Is(err, errBypassProxy) {
				viaExistingDial = false
				bypassed = true
				conn, err = nil, nil
			}
		}

		switch {
		case viaExistingDial:
		case existingDialTimeout != nil && !bypassed:
			// A bypassed target never takes this callback: it has no IP checks.
			// Preserve dial-timeout behavior
			conn, err = existingDialTimeout(addr, client.ReadTimeout)
		default:
			// resolve DNS ourselves, reject private IPs, then dial
			// the IP literal directly — closes the DNS rebinding window that exists
			// between ValidateExternalURL (save time) and this connection.
			host, port, splitErr := net.SplitHostPort(addr)
			if splitErr != nil {
				return nil, splitErr
			}
			// Bound DNS resolution with the same timeout that governs the TCP
			resolveCtx := context.Background()
			if client.ReadTimeout > 0 {
				var cancel context.CancelFunc
				resolveCtx, cancel = context.WithTimeout(resolveCtx, client.ReadTimeout)
				defer cancel()
			}
			ips, resolveErr := net.DefaultResolver.LookupIP(resolveCtx, "ip", host)
			if resolveErr != nil {
				return nil, resolveErr
			}
			dialer := &net.Dialer{
				Timeout:         client.ReadTimeout,
				KeepAliveConfig: keepAliveCfg,
			}
			var lastErr error
			for _, ip := range ips {
				// Unspecified (0.0.0.0, ::) and link-local (169.254.x.x, fe80::) are always blocked
				if ip.IsUnspecified() {
					return nil, fmt.Errorf("connection to unspecified IP %s is not allowed", ip)
				}
				if network.IsLinkLocal(ip) {
					return nil, fmt.Errorf("connection to link-local IP %s is not allowed", ip)
				}
				// RFC 1918 blocked unless operator explicitly opted in; loopback always allowed
				if !ip.IsLoopback() && !allowPrivateNetwork && network.IsPrivateIP(ip) {
					return nil, fmt.Errorf("connection to private IP %s is not allowed", ip)
				}
				conn, err = dialer.Dial("tcp", net.JoinHostPort(ip.String(), port))
				if err == nil {
					break
				}
				lastErr = err
			}
			if conn == nil {
				if lastErr != nil {
					return nil, lastErr
				}
				return nil, fmt.Errorf("no usable address resolved for %s", host)
			}
		}
		if err != nil {
			return nil, err
		}

		// Enable TCP keepalive on the connection
		if tcpConn, ok := conn.(*net.TCPConn); ok {
			_ = tcpConn.SetKeepAliveConfig(keepAliveCfg)
		}
		return conn, nil
	}

	return client
}
