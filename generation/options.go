// Package generation defines shared generation controls for native and hosted models.
package generation

import (
	"fmt"
	"math"
	"strings"
	"unicode/utf8"
)

type Config struct {
	MaxTokens   int
	Temperature float64
	Seed        int64
	Stop        []string
	OnToken     func(string) error
}
type Option func(*Config)

func WithMaxTokens(n int) Option { return func(c *Config) { c.MaxTokens = n } }

// WithTemperature(0) selects greedy decoding for native models.
func WithTemperature(t float64) Option { return func(c *Config) { c.Temperature = t } }
func WithSeed(seed int64) Option       { return func(c *Config) { c.Seed = seed } }
func WithStop(stop ...string) Option {
	stop = append([]string(nil), stop...)
	return func(c *Config) { c.Stop = append([]string(nil), stop...) }
}

// WithStream receives generated chunks synchronously. Returning an error cancels
// generation. Callbacks must not re-enter a native model that is generating.
func WithStream(f func(string) error) Option { return func(c *Config) { c.OnToken = f } }
func Resolve(options []Option) (Config, error) {
	c := Config{MaxTokens: 128, Temperature: 0.8, Seed: 1}
	for _, option := range options {
		if option == nil {
			return c, fmt.Errorf("gofish: nil generation option")
		}
		option(&c)
	}
	if c.MaxTokens < 1 || c.MaxTokens > 1000000 {
		return c, fmt.Errorf("gofish: MaxTokens must be between 1 and 1000000")
	}
	if c.Temperature < 0 || math.IsNaN(c.Temperature) || math.IsInf(c.Temperature, 0) {
		return c, fmt.Errorf("gofish: temperature must be finite and nonnegative")
	}
	for _, s := range c.Stop {
		if s == "" {
			return c, fmt.Errorf("gofish: stop sequences must be nonempty")
		}
	}
	return c, nil
}

// Collector buffers partial stop sequences and UTF-8 characters, so callbacks
// never receive a stop sequence, even when it spans multiple streamed chunks.
type Collector struct {
	config  Config
	pending string
	text    strings.Builder
	stopped bool
	hold    int
}

func NewCollector(c Config) *Collector {
	s := &Collector{config: c}
	for _, stop := range c.Stop {
		s.hold = max(s.hold, len(stop)-1)
	}
	return s
}
func (s *Collector) emit(text string) error {
	if text == "" {
		return nil
	}
	s.text.WriteString(text)
	if s.config.OnToken != nil {
		return s.config.OnToken(text)
	}
	return nil
}
func (s *Collector) Feed(chunk string) (bool, error) {
	if s.stopped {
		return true, nil
	}
	s.pending += chunk
	end := -1
	for _, stop := range s.config.Stop {
		if i := strings.Index(s.pending, stop); i >= 0 && (end < 0 || i < end) {
			end = i
		}
	}
	if end >= 0 {
		s.stopped = true
		err := s.emit(s.pending[:end])
		s.pending = ""
		return true, err
	}
	n := max(0, len(s.pending)-s.hold)
	for n > 0 && (!utf8.ValidString(s.pending[:n])) {
		n--
	}
	err := s.emit(s.pending[:n])
	s.pending = s.pending[n:]
	return false, err
}
func (s *Collector) Finish() error {
	if s.stopped {
		return nil
	}
	err := s.emit(s.pending)
	s.pending = ""
	return err
}
func (s *Collector) Text() string { return s.text.String() }
