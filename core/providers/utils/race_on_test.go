//go:build race

package utils

// raceEnabled: the race detector randomly drops sync.Pool items, so
// allocation counts that rely on a pooled buffer are not stable under it.
const raceEnabled = true
