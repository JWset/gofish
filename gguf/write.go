package gguf

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"sort"
	"unicode/utf8"
)

type TensorData struct {
	Name       string
	Dimensions []uint64
	Type       Type
	Data       []byte
}

func writeValue(w io.Writer, value any, withType bool) error {
	var kind uint32
	switch value.(type) {
	case uint8:
		kind = 0
	case int8:
		kind = 1
	case uint16:
		kind = 2
	case int16:
		kind = 3
	case uint32:
		kind = 4
	case int32:
		kind = 5
	case float32:
		kind = 6
	case bool:
		kind = 7
	case string:
		kind = 8
	case []string, []float32, []int32:
		kind = 9
	case uint64:
		kind = 10
	case int64:
		kind = 11
	case float64:
		kind = 12
	default:
		return fmt.Errorf("gofish: unsupported GGUF metadata value %T", value)
	}
	if withType {
		if err := binary.Write(w, binary.LittleEndian, kind); err != nil {
			return err
		}
	}
	switch v := value.(type) {
	case string:
		if !utf8.ValidString(v) {
			return fmt.Errorf("gofish: invalid UTF-8 GGUF string")
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		_, err := io.WriteString(w, v)
		return err
	case bool:
		b := uint8(0)
		if v {
			b = 1
		}
		return binary.Write(w, binary.LittleEndian, b)
	case []string:
		if err := binary.Write(w, binary.LittleEndian, uint32(8)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		for _, s := range v {
			if err := writeValue(w, s, false); err != nil {
				return err
			}
		}
		return nil
	case []float32:
		if err := binary.Write(w, binary.LittleEndian, uint32(6)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, v)
	case []int32:
		if err := binary.Write(w, binary.LittleEndian, uint32(5)); err != nil {
			return err
		}
		if err := binary.Write(w, binary.LittleEndian, uint64(len(v))); err != nil {
			return err
		}
		return binary.Write(w, binary.LittleEndian, v)
	default:
		return binary.Write(w, binary.LittleEndian, value)
	}
}

// Write emits a deterministic GGUF v3 file with 32-byte alignment. It does not
// quantize tensors: provide an already encoded payload matching its declared type.
func Write(w io.Writer, metadata map[string]any, tensors []TensorData) error {
	if a, ok := metadata["general.alignment"]; ok && a != uint32(32) {
		return fmt.Errorf("gofish: writer requires general.alignment uint32(32)")
	}
	var header bytes.Buffer
	for _, value := range []any{uint32(0x46554747), uint32(3), uint64(len(tensors)), uint64(len(metadata))} {
		if err := binary.Write(&header, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	keys := make([]string, 0, len(metadata))
	for key := range metadata {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		if err := writeValue(&header, key, false); err != nil {
			return err
		}
		if err := writeValue(&header, metadata[key], true); err != nil {
			return err
		}
	}
	offset := uint64(0)
	names := map[string]bool{}
	for _, t := range tensors {
		if t.Name == "" || names[t.Name] || len(t.Dimensions) < 1 || len(t.Dimensions) > 4 {
			return fmt.Errorf("gofish: invalid tensor descriptor")
		}
		names[t.Name] = true
		elements := uint64(1)
		for _, d := range t.Dimensions {
			if d == 0 || d > math.MaxInt64/elements {
				return fmt.Errorf("gofish: invalid tensor shape")
			}
			elements *= d
		}
		block, bytes, ok := layout(t.Type)
		if !ok || t.Dimensions[0]%block != 0 || elements/block > math.MaxInt64/bytes || uint64(len(t.Data)) != elements/block*bytes {
			return fmt.Errorf("gofish: tensor %q payload does not match shape and encoding", t.Name)
		}
		if err := writeValue(&header, t.Name, false); err != nil {
			return err
		}
		for _, value := range []any{uint32(len(t.Dimensions)), t.Dimensions, uint32(t.Type), offset} {
			if err := binary.Write(&header, binary.LittleEndian, value); err != nil {
				return err
			}
		}
		offset = (offset + uint64(len(t.Data)) + 31) &^ 31
	}
	padding := make([]byte, 32)
	if _, err := w.Write(header.Bytes()); err != nil {
		return err
	}
	if n := (32 - header.Len()%32) % 32; n > 0 {
		if _, err := w.Write(padding[:n]); err != nil {
			return err
		}
	}
	for _, t := range tensors {
		if _, err := w.Write(t.Data); err != nil {
			return err
		}
		if n := (32 - len(t.Data)%32) % 32; n > 0 {
			if _, err := w.Write(padding[:n]); err != nil {
				return err
			}
		}
	}
	return nil
}
