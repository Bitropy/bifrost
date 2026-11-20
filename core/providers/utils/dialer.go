package utils

import (
	"context"
	"errors"
	"fmt"
	"net"
	"time"

	"github.com/maximhq/bifrost/core/network"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/valyala/fasthttp"
)

// ErrConnectTimeout marks a DNS resolution or TCP dial that ran out of its
// connect budget, so classification can tell an unreachable endpoint from a
// slow answer. The wrapped error names the address and keeps the cause.
var ErrConnectTimeout = errors.New("connect timeout")

// wrapConnectTimeout tags timeout-flavored dial/resolve errors with
// ErrConnectTimeout; other errors pass through unchanged.
func wrapConnectTimeout(err error, addr string) error {
	if err == nil {
		return nil
	}
	var netErr net.Error
	if (errors.As(err, &netErr) && netErr.Timeout()) || errors.Is(err, context.DeadlineExceeded) {
		return fmt.Errorf("%w dialing %s: %w", ErrConnectTimeout, addr, err)
	}
	return err
}

// connectBudget is the one rule for the DNS + dial budget:
// ConnectTimeoutInSeconds, never longer than a set requestTimeout; when
// ConnectTimeoutInSeconds is unset (CheckAndSetDefaults not run), requestTimeout.
func connectBudget(nc schemas.NetworkConfig, requestTimeout time.Duration) time.Duration {
	connectTimeout := time.Duration(nc.ConnectTimeoutInSeconds) * time.Second
	if connectTimeout <= 0 || (requestTimeout > 0 && requestTimeout < connectTimeout) {
		return requestTimeout
	}
	return connectTimeout
}

// ConnectTimeoutDialContext is a net/http DialContext that only adds the
// connect budget and the ErrConnectTimeout tag to a plain net.Dialer. Unlike
// ConfigureDialer it runs no IP checks (no private, link-local or unspecified
// address filtering) and keeps net.Dialer's default keepalive; it is for
// net/http providers (Bedrock) whose previous stdlib dialer had neither.
func ConnectTimeoutDialContext(nc schemas.NetworkConfig, requestTimeout time.Duration) func(ctx context.Context, network, addr string) (net.Conn, error) {
	dialer := &net.Dialer{Timeout: connectBudget(nc, requestTimeout), KeepAlive: 30 * time.Second}
	return func(ctx context.Context, network, addr string) (net.Conn, error) {
		conn, err := dialer.DialContext(ctx, network, addr)
		return conn, wrapConnectTimeout(err, addr)
	}
}

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
//
// DNS resolution and the TCP dial of every resolved address share one budget,
// connectBudget(networkConfig, client.ReadTimeout), so an unreachable endpoint
// (e.g. a base_url behind a packet-dropping firewall) fails fast instead of
// holding the request for the full request timeout. A client whose Dial was set
// before this call (ConfigureProxy's HTTP/SOCKS5 proxy dialers) keeps that
// dialer's own bound; a preset DialTimeout callback receives the connect budget.
// Every path's timeout is tagged with ErrConnectTimeout. The budget reads
// client.ReadTimeout once, so set ReadTimeout before calling ConfigureDialer.
func ConfigureDialer(client *fasthttp.Client, networkConfig schemas.NetworkConfig) *fasthttp.Client {
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

	allowPrivateNetwork := networkConfig.AllowPrivateNetwork
	connectTimeout := connectBudget(networkConfig, client.ReadTimeout)

	existingDial := client.Dial
	existingDialTimeout := client.DialTimeout

	keepAliveCfg := net.KeepAliveConfig{
		Enable:   true,
		Idle:     10 * time.Second,
		Interval: 5 * time.Second,
		Count:    3,
	}

	dial := func(addr string) (net.Conn, error) {
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
			conn, err = existingDialTimeout(addr, connectTimeout)
		default:
			// resolve DNS ourselves, reject private IPs, then dial
			// the IP literal directly — closes the DNS rebinding window that exists
			// between ValidateExternalURL (save time) and this connection.
			host, port, splitErr := net.SplitHostPort(addr)
			if splitErr != nil {
				return nil, splitErr
			}
			// One connect budget covers DNS resolution and the dials of every resolved
			// IP together (dialResolved splits what is left between the addresses).
			resolveCtx := context.Background()
			if connectTimeout > 0 {
				var cancel context.CancelFunc
				resolveCtx, cancel = context.WithTimeout(resolveCtx, connectTimeout)
				defer cancel()
			}
			ips, resolveErr := net.DefaultResolver.LookupIP(resolveCtx, "ip", host)
			if resolveErr != nil {
				return nil, resolveErr
			}
			dialer := &net.Dialer{KeepAliveConfig: keepAliveCfg}
			conn, err = dialResolved(resolveCtx, ips, port, func(ip net.IP) error {
				// Unspecified (0.0.0.0, ::) and link-local (169.254.x.x, fe80::) are always blocked
				if ip.IsUnspecified() {
					return fmt.Errorf("connection to unspecified IP %s is not allowed", ip)
				}
				if network.IsLinkLocal(ip) {
					return fmt.Errorf("connection to link-local IP %s is not allowed", ip)
				}
				// RFC 1918 blocked unless operator explicitly opted in; loopback always allowed
				if !ip.IsLoopback() && !allowPrivateNetwork && network.IsPrivateIP(ip) {
					return fmt.Errorf("connection to private IP %s is not allowed", ip)
				}
				return nil
			}, dialer.DialContext)
			if conn == nil && err == nil {
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
	// Tag every path's connect timeout once, here: wrapConnectTimeout passes any
	// other error (blocked IP, proxy refusal) through unchanged.
	client.Dial = func(addr string) (net.Conn, error) {
		conn, err := dial(addr)
		return conn, wrapConnectTimeout(err, addr)
	}

	return client
}

// dialResolved dials the resolved addresses in order until one connects. Like
// net.Dialer's own fallback, each attempt gets an equal share of the time left
// on ctx, so one black-holed address (e.g. a broken IPv6 route) cannot use up
// the whole connect budget before a working one is tried. check rejects an
// address before it is dialed and ends the attempt. No addresses: (nil, nil).
func dialResolved(ctx context.Context, ips []net.IP, port string, check func(net.IP) error, dial func(ctx context.Context, network, addr string) (net.Conn, error)) (net.Conn, error) {
	var lastErr error
	for i, ip := range ips {
		if err := check(ip); err != nil {
			return nil, err
		}
		attemptCtx, cancel := ctx, context.CancelFunc(func() {})
		if deadline, ok := ctx.Deadline(); ok {
			attemptCtx, cancel = context.WithTimeout(ctx, time.Until(deadline)/time.Duration(len(ips)-i))
		}
		conn, err := dial(attemptCtx, "tcp", net.JoinHostPort(ip.String(), port))
		cancel()
		if err == nil {
			return conn, nil
		}
		lastErr = err
	}
	return nil, lastErr
}
