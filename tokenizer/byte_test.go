package tokenizer_test

import (
	"testing"

	"github.com/JWset/gofish/tokenizer"
)

func TestLosslessUnicode(t *testing.T) {
	codec := tokenizer.Byte{}
	for _, text := range []string{"", "hello\x00world", "你好 🐟 café", "\ufffd"} {
		ids := codec.Encode(text)
		decoded, err := codec.Decode(ids)
		if err != nil || decoded != text {
			t.Fatalf("%q -> %q (%v)", text, decoded, err)
		}
	}
	for _, ids := range [][]int{{256}, {-1}, {0xf0, 0x9f}, {0xc0, 0x80}} {
		if _, err := codec.Decode(ids); err == nil {
			t.Fatal("invalid sequence accepted", ids)
		}
	}
}
func TestUTF8Prefixes(t *testing.T) {
	for _, tt := range []struct {
		prefix []byte
		next   byte
		valid  bool
	}{
		{nil, 'a', true}, {nil, 0xf0, true}, {nil, 0x80, false}, {nil, 0xc0, false}, {[]byte{0xf0}, 0x9f, true}, {[]byte{0xf0}, 0x80, false}, {[]byte{0xed}, 0xa0, false}, {[]byte{0xf4}, 0x90, false}, {[]byte{0xf0, 0x9f, 0x90}, 0x9f, true},
	} {
		if got := tokenizer.ValidNext(tt.prefix, tt.next); got != tt.valid {
			t.Fatalf("%x + %x: %v", tt.prefix, tt.next, got)
		}
	}
}
