package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

func half(bits uint16) float32 {
	sign := uint32(bits&0x8000) << 16
	exp, mantissa := (bits>>10)&31, bits&1023
	if exp == 0 {
		v := float32(math.Ldexp(float64(mantissa), -24))
		if sign != 0 {
			v = -v
		}
		return v
	}
	if exp == 31 {
		return math.Float32frombits(sign | 0x7f800000 | uint32(mantissa)<<13)
	}
	return math.Float32frombits(sign | uint32(exp+112)<<23 | uint32(mantissa)<<13)
}
func h(data []byte) float32 { return half(binary.LittleEndian.Uint16(data)) }

// Decode expands one contiguous tensor payload. Block encodings require an
// element count divisible by their block size; malformed lengths return errors.
func Decode(kind Type, data []byte, elements int) ([]float32, error) {
	block, bytes, ok := layout(kind)
	if !ok {
		return nil, fmt.Errorf("gofish: unsupported GGUF encoding %s", kind)
	}
	if elements < 0 || uint64(elements)%block != 0 || uint64(elements)/block*bytes != uint64(len(data)) {
		return nil, fmt.Errorf("gofish: invalid %s tensor byte length", kind)
	}
	out := make([]float32, elements)
	for start, offset := 0, 0; start < elements; start, offset = start+int(block), offset+int(bytes) {
		decodeBlock(kind, data[offset:offset+int(bytes)], out[start:start+int(block)])
	}
	return out, nil
}
func scaleMin(scales []byte, index int) (float32, float32) {
	if index < 4 {
		return float32(scales[index] & 63), float32(scales[index+4] & 63)
	}
	return float32((scales[index+4] & 15) | ((scales[index-4] >> 6) << 4)), float32((scales[index+4] >> 4) | ((scales[index] >> 6) << 4))
}
func decodeBlock(kind Type, data []byte, out []float32) {
	switch kind {
	case F32:
		out[0] = math.Float32frombits(binary.LittleEndian.Uint32(data))
	case F16:
		out[0] = h(data)
	case BF16:
		out[0] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(data)) << 16)
	case Q8_0:
		d := h(data)
		for i := 0; i < 32; i++ {
			out[i] = d * float32(int8(data[2+i]))
		}
	case Q4_0, Q4_1, Q5_0, Q5_1:
		d, minimum, offset := h(data), float32(0), 2
		bits := 4
		if kind == Q4_1 || kind == Q5_1 {
			minimum = h(data[2:])
			offset = 4
		}
		var high uint32
		if kind == Q5_0 || kind == Q5_1 {
			bits = 5
			high = binary.LittleEndian.Uint32(data[offset:])
			offset += 4
		}
		for i := 0; i < 32; i++ {
			q := int(data[offset+i%16])
			if i < 16 {
				q &= 15
			} else {
				q >>= 4
			}
			if bits == 5 {
				q |= int((high>>i)&1) << 4
			}
			if kind == Q4_0 {
				q -= 8
			}
			if kind == Q5_0 {
				q -= 16
			}
			out[i] = d*float32(q) + minimum
		}
	case Q4_K:
		d, minimum := h(data), h(data[2:])
		for group := 0; group < 8; group++ {
			scale, minScale := scaleMin(data[4:16], group)
			for j := 0; j < 32; j++ {
				q := data[16+(group/2)*32+j]
				if group%2 == 0 {
					q &= 15
				} else {
					q >>= 4
				}
				out[group*32+j] = d*scale*float32(q) - minimum*minScale
			}
		}
	case Q6_K:
		d := h(data[208:])
		for i := 0; i < 256; i++ {
			part, local := i/128, i%128
			group, lane := local/32, local%32
			low := data[part*64+(group%2)*32+lane]
			if group < 2 {
				low &= 15
			} else {
				low >>= 4
			}
			high := ((data[128+part*32+lane] >> uint(2*group)) & 3) << 4
			scale := int8(data[192+part*8+group*2+lane/16])
			out[i] = d * float32(scale) * float32(int(low|high)-32)
		}
	}
}

// Matrix keeps weights in their original encoding. MatVec decodes blocks into a
// fixed scratch buffer instead of expanding the whole model to floating point.
type Matrix struct {
	Rows, Columns int
	Type          Type
	data          []byte
}

// NewPackedMatrix owns a copy of an already encoded row-major matrix.
func NewPackedMatrix(rows, columns int, kind Type, data []byte) (*Matrix, error) {
	block, bytes, ok := layout(kind)
	if !ok || rows < 1 || columns < 1 || uint64(columns)%block != 0 || uint64(rows) > math.MaxInt64/uint64(columns) || uint64(rows)*uint64(columns)/block > math.MaxInt64/bytes || uint64(len(data)) != uint64(rows)*uint64(columns)/block*bytes {
		return nil, fmt.Errorf("gofish: invalid packed matrix")
	}
	return &Matrix{Rows: rows, Columns: columns, Type: kind, data: append([]byte(nil), data...)}, nil
}

func NewMatrix(t Tensor, maxBytes uint64) (*Matrix, error) {
	if len(t.Dimensions) != 2 || t.Dimensions[0] > uint64(int(^uint(0)>>1)) || t.Dimensions[1] > uint64(int(^uint(0)>>1)) {
		return nil, fmt.Errorf("gofish: tensor %q is not a supported matrix", t.Name)
	}
	data, err := t.Bytes(maxBytes)
	if err != nil {
		return nil, err
	}
	return &Matrix{Rows: int(t.Dimensions[1]), Columns: int(t.Dimensions[0]), Type: t.Type, data: data}, nil
}
func (m *Matrix) Row(row int, dst []float32) error {
	if row < 0 || row >= m.Rows || len(dst) != m.Columns {
		return fmt.Errorf("gofish: matrix row shape mismatch")
	}
	block, bytes, _ := layout(m.Type)
	start := row * m.Columns / int(block) * int(bytes)
	for col, offset := 0, start; col < m.Columns; col, offset = col+int(block), offset+int(bytes) {
		decodeBlock(m.Type, m.data[offset:offset+int(bytes)], dst[col:col+int(block)])
	}
	return nil
}
func (m *Matrix) MatVec(input []float32) ([]float32, error) {
	if len(input) != m.Columns {
		return nil, fmt.Errorf("gofish: matrix vector shape mismatch")
	}
	block, bytes, _ := layout(m.Type)
	out := make([]float32, m.Rows)
	var scratch [256]float32
	for row := range out {
		sum := float64(0)
		start := row * m.Columns / int(block) * int(bytes)
		for col, offset := 0, start; col < m.Columns; col, offset = col+int(block), offset+int(bytes) {
			decodeBlock(m.Type, m.data[offset:offset+int(bytes)], scratch[:int(block)])
			for j := 0; j < int(block); j++ {
				sum += float64(scratch[j]) * float64(input[col+j])
			}
		}
		out[row] = float32(sum)
	}
	return out, nil
}
func (m *Matrix) Values(maxElements int) ([]float32, error) {
	if maxElements < 0 || m.Columns < 1 || m.Rows < 1 || m.Rows > maxElements/m.Columns {
		return nil, fmt.Errorf("gofish: matrix exceeds element budget")
	}
	return Decode(m.Type, m.data, m.Rows*m.Columns)
}

// Float32Data encodes ordinary row-major float32 weights for a GGUF writer.
func Float32Data(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
