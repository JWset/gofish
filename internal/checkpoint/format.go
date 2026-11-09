// Package checkpoint dispatches native formats without decoding weight arrays twice.
package checkpoint

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Format reads bounded JSON metadata. Save implementations put format first.
func Format(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	z, err := gzip.NewReader(f)
	if err != nil {
		return "", err
	}
	defer z.Close()
	decoder := json.NewDecoder(io.LimitReader(z, 256<<20))
	token, err := decoder.Token()
	if err != nil {
		return "", err
	}
	if token != json.Delim('{') {
		return "", fmt.Errorf("gofish: invalid checkpoint metadata")
	}
	for decoder.More() {
		key, err := decoder.Token()
		if err != nil {
			return "", err
		}
		if key == "format" {
			var format string
			err = decoder.Decode(&format)
			return format, err
		}
		var skip json.RawMessage
		if err = decoder.Decode(&skip); err != nil {
			return "", err
		}
	}
	return "", fmt.Errorf("gofish: checkpoint has no format metadata")
}
