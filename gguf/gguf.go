// Package gguf reads and writes little-endian GGUF v2/v3 model files. Tensor
// payloads are read lazily, so inspecting a model does not load its weights.
package gguf

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"unicode/utf8"
)

type Type uint32

const (
	F32  Type = 0
	F16  Type = 1
	Q4_0 Type = 2
	Q4_1 Type = 3
	Q5_0 Type = 6
	Q5_1 Type = 7
	Q8_0 Type = 8
	Q4_K Type = 12
	Q6_K Type = 14
	BF16 Type = 30
)

func (t Type) String() string {
	if name, ok := map[Type]string{F32: "F32", F16: "F16", Q4_0: "Q4_0", Q4_1: "Q4_1", Q5_0: "Q5_0", Q5_1: "Q5_1", Q8_0: "Q8_0", Q4_K: "Q4_K", Q6_K: "Q6_K", BF16: "BF16"}[t]; ok {
		return name
	}
	return fmt.Sprintf("type(%d)", t)
}
func layout(t Type) (block, bytes uint64, ok bool) {
	switch t {
	case F32:
		return 1, 4, true
	case F16, BF16:
		return 1, 2, true
	case Q4_0:
		return 32, 18, true
	case Q4_1:
		return 32, 20, true
	case Q5_0:
		return 32, 22, true
	case Q5_1:
		return 32, 24, true
	case Q8_0:
		return 32, 34, true
	case Q4_K:
		return 256, 144, true
	case Q6_K:
		return 256, 210, true
	}
	return 0, 0, false
}

type Tensor struct {
	Name       string
	Dimensions []uint64 // GGML order: the contiguous column dimension comes first.
	Type       Type
	Offset     uint64 // Relative to the tensor-data section.
	Size       uint64
	file       *File
}

func (t Tensor) Elements() uint64 {
	n := uint64(1)
	for _, d := range t.Dimensions {
		n *= d
	}
	return n
}

type File struct {
	Version    uint32
	Metadata   map[string]any
	Tensors    []Tensor
	DataOffset int64
	reader     io.ReaderAt
	owned      *os.File
}

func Open(path string) (*File, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}
	model, err := Read(f, info.Size())
	if err != nil {
		f.Close()
		return nil, err
	}
	model.owned = f
	return model, nil
}
func (f *File) Close() error {
	if f.owned != nil {
		return f.owned.Close()
	}
	return nil
}
func (f *File) Tensor(name string) (Tensor, bool) {
	for _, t := range f.Tensors {
		if t.Name == name {
			return t, true
		}
	}
	return Tensor{}, false
}
func (f *File) Uint(key string, defaultValue uint64) uint64 {
	v := f.Metadata[key]
	switch x := v.(type) {
	case uint8:
		return uint64(x)
	case uint16:
		return uint64(x)
	case uint32:
		return uint64(x)
	case uint64:
		return x
	case int8:
		if x >= 0 {
			return uint64(x)
		}
	case int16:
		if x >= 0 {
			return uint64(x)
		}
	case int32:
		if x >= 0 {
			return uint64(x)
		}
	case int64:
		if x >= 0 {
			return uint64(x)
		}
	}
	return defaultValue
}
func (f *File) Float(key string, defaultValue float64) float64 {
	switch x := f.Metadata[key].(type) {
	case float32:
		return float64(x)
	case float64:
		return x
	}
	return defaultValue
}
func (f *File) String(key, defaultValue string) string {
	if s, ok := f.Metadata[key].(string); ok {
		return s
	}
	return defaultValue
}
func (f *File) Bool(key string, defaultValue bool) bool {
	if b, ok := f.Metadata[key].(bool); ok {
		return b
	}
	return defaultValue
}

// Bytes reads a supported tensor subject to a caller-supplied byte budget.
func (t Tensor) Bytes(maxBytes uint64) ([]byte, error) {
	if _, _, ok := layout(t.Type); !ok {
		return nil, fmt.Errorf("gofish: unsupported GGUF tensor encoding %s", t.Type)
	}
	if t.file == nil || t.Size > maxBytes || t.Size > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("gofish: tensor %q exceeds byte budget", t.Name)
	}
	data := make([]byte, int(t.Size))
	_, err := t.file.reader.ReadAt(data, t.file.DataOffset+int64(t.Offset))
	if err != nil {
		return nil, err
	}
	return data, nil
}
func (t Tensor) Decode(maxElements uint64) ([]float32, error) {
	if t.Elements() > maxElements {
		return nil, fmt.Errorf("gofish: tensor %q exceeds element budget", t.Name)
	}
	data, err := t.Bytes(maxElements * 4)
	if err != nil {
		return nil, err
	}
	return Decode(t.Type, data, int(t.Elements()))
}

type parser struct {
	r    io.Reader
	n    int64
	size int64
}

func (p *parser) Read(data []byte) (int, error) {
	if int64(len(data)) > 64<<20-p.n {
		return 0, fmt.Errorf("gofish: GGUF metadata exceeds 64 MiB")
	}
	n, err := p.r.Read(data)
	p.n += int64(n)
	return n, err
}
func read[T any](p *parser) (T, error) {
	var value T
	err := binary.Read(p, binary.LittleEndian, &value)
	return value, err
}
func (p *parser) str() (string, error) {
	n, err := read[uint64](p)
	if err != nil {
		return "", err
	}
	if n > 16<<20 || n > uint64(p.size-p.n) {
		return "", fmt.Errorf("gofish: invalid GGUF string length")
	}
	data := make([]byte, int(n))
	if _, err = io.ReadFull(p, data); err != nil {
		return "", err
	}
	if !utf8.Valid(data) {
		return "", fmt.Errorf("gofish: GGUF strings must be UTF-8")
	}
	return string(data), nil
}
func (p *parser) value(kind uint32) (any, error) {
	switch kind {
	case 0:
		return read[uint8](p)
	case 1:
		return read[int8](p)
	case 2:
		return read[uint16](p)
	case 3:
		return read[int16](p)
	case 4:
		return read[uint32](p)
	case 5:
		return read[int32](p)
	case 6:
		return read[float32](p)
	case 7:
		b, err := read[uint8](p)
		if err != nil {
			return nil, err
		}
		if b > 1 {
			return nil, fmt.Errorf("gofish: invalid GGUF boolean")
		}
		return b == 1, nil
	case 8:
		return p.str()
	case 9:
		element, err := read[uint32](p)
		if err != nil {
			return nil, err
		}
		n, err := read[uint64](p)
		if err != nil {
			return nil, err
		}
		if element > 12 || element == 9 || n > 1000000 || n > uint64(p.size-p.n) {
			return nil, fmt.Errorf("gofish: invalid or nested GGUF array")
		}
		if element == 8 {
			out := make([]string, int(n))
			for i := range out {
				out[i], err = p.str()
				if err != nil {
					return nil, err
				}
			}
			return out, nil
		}
		if element == 6 {
			out := make([]float32, int(n))
			for i := range out {
				out[i], err = read[float32](p)
				if err != nil {
					return nil, err
				}
			}
			return out, nil
		}
		if element == 5 {
			out := make([]int32, int(n))
			for i := range out {
				out[i], err = read[int32](p)
				if err != nil {
					return nil, err
				}
			}
			return out, nil
		}
		out := make([]any, int(n))
		for i := range out {
			out[i], err = p.value(element)
			if err != nil {
				return nil, err
			}
		}
		return out, nil
	case 10:
		return read[uint64](p)
	case 11:
		return read[int64](p)
	case 12:
		return read[float64](p)
	}
	return nil, fmt.Errorf("gofish: unknown GGUF metadata type %d", kind)
}

// Read validates headers, metadata, shapes, alignment, tensor extents and overlap.
// Unknown tensor encodings remain inspectable, but cannot be decoded.
func Read(reader io.ReaderAt, size int64) (*File, error) {
	if reader == nil || size < 24 {
		return nil, fmt.Errorf("gofish: truncated GGUF header")
	}
	p := &parser{r: io.NewSectionReader(reader, 0, size), size: size}
	magic, err := read[uint32](p)
	if err != nil {
		return nil, err
	}
	if magic != 0x46554747 {
		return nil, fmt.Errorf("gofish: expected little-endian GGUF magic")
	}
	version, err := read[uint32](p)
	if err != nil {
		return nil, err
	}
	if version != 2 && version != 3 {
		return nil, fmt.Errorf("gofish: unsupported GGUF version %d", version)
	}
	nt, err := read[uint64](p)
	if err != nil {
		return nil, err
	}
	nk, err := read[uint64](p)
	if err != nil {
		return nil, err
	}
	if nt > 100000 || nk > 100000 || nt > uint64(size)/24 || nk > uint64(size)/12 {
		return nil, fmt.Errorf("gofish: excessive GGUF header counts")
	}
	f := &File{Version: version, Metadata: map[string]any{}, reader: reader}
	for i := uint64(0); i < nk; i++ {
		key, err := p.str()
		if err != nil {
			return nil, err
		}
		if _, exists := f.Metadata[key]; exists {
			return nil, fmt.Errorf("gofish: duplicate GGUF key %q", key)
		}
		kind, err := read[uint32](p)
		if err != nil {
			return nil, err
		}
		value, err := p.value(kind)
		if err != nil {
			return nil, err
		}
		f.Metadata[key] = value
	}
	alignment := f.Uint("general.alignment", 32)
	if value, exists := f.Metadata["general.alignment"]; exists {
		if _, ok := value.(uint32); !ok {
			return nil, fmt.Errorf("gofish: GGUF alignment must be uint32")
		}
	}
	if alignment < 1 || alignment > 4096 || alignment&(alignment-1) != 0 {
		return nil, fmt.Errorf("gofish: invalid GGUF alignment")
	}
	names := map[string]bool{}
	for i := uint64(0); i < nt; i++ {
		name, err := p.str()
		if err != nil {
			return nil, err
		}
		if name == "" || names[name] {
			return nil, fmt.Errorf("gofish: empty or duplicate GGUF tensor name")
		}
		names[name] = true
		dimensions, err := read[uint32](p)
		if err != nil {
			return nil, err
		}
		if dimensions < 1 || dimensions > 4 {
			return nil, fmt.Errorf("gofish: invalid GGUF tensor rank")
		}
		t := Tensor{Name: name, Dimensions: make([]uint64, int(dimensions)), file: f}
		elements := uint64(1)
		for j := range t.Dimensions {
			d, err := read[uint64](p)
			if err != nil {
				return nil, err
			}
			if d == 0 || d > math.MaxInt64/elements {
				return nil, fmt.Errorf("gofish: invalid GGUF tensor dimensions")
			}
			elements *= d
			t.Dimensions[j] = d
		}
		kind, err := read[uint32](p)
		if err != nil {
			return nil, err
		}
		t.Type = Type(kind)
		t.Offset, err = read[uint64](p)
		if err != nil {
			return nil, err
		}
		if t.Offset%alignment != 0 || t.Offset > uint64(size) {
			return nil, fmt.Errorf("gofish: invalid GGUF tensor offset")
		}
		if block, bytes, ok := layout(t.Type); ok {
			if t.Dimensions[0]%block != 0 || elements/block > math.MaxInt64/bytes {
				return nil, fmt.Errorf("gofish: invalid GGUF quantization block shape")
			}
			t.Size = elements / block * bytes
		}
		f.Tensors = append(f.Tensors, t)
	}
	f.DataOffset = int64((uint64(p.n) + alignment - 1) &^ (alignment - 1))
	if f.DataOffset > size {
		return nil, fmt.Errorf("gofish: missing GGUF data section")
	}
	ordered := append([]Tensor(nil), f.Tensors...)
	sort.Slice(ordered, func(i, j int) bool { return ordered[i].Offset < ordered[j].Offset })
	for i, t := range ordered {
		if t.Offset > uint64(size-f.DataOffset) || t.Size > uint64(size-f.DataOffset)-t.Offset {
			return nil, fmt.Errorf("gofish: truncated tensor %q", t.Name)
		}
		if i+1 < len(ordered) && t.Offset+t.Size > ordered[i+1].Offset {
			return nil, fmt.Errorf("gofish: overlapping GGUF tensors")
		}
	}
	return f, nil
}
