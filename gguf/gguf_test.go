package gguf

import (
	"bytes"
	"encoding/binary"
	"math"
	"testing"
)

func TestRoundTripAndPackedMatVec(t *testing.T) {
	metadata := map[string]any{"general.architecture": "llama", "test.uint": uint32(7), "test.bool": true, "test.strings": []string{"hello", "🐟"}, "test.scores": []float32{1, 2}, "test.ids": []int32{3, 4}}
	values := []float32{1, 2, 3, 4, 5, 6}
	var buffer bytes.Buffer
	if err := Write(&buffer, metadata, []TensorData{{Name: "weight", Dimensions: []uint64{3, 2}, Type: F32, Data: Float32Data(values)}}); err != nil {
		t.Fatal(err)
	}
	f, err := Read(bytes.NewReader(buffer.Bytes()), int64(buffer.Len()))
	if err != nil {
		t.Fatal(err)
	}
	if f.Uint("test.uint", 0) != 7 || !f.Bool("test.bool", false) || f.DataOffset%32 != 0 {
		t.Fatal("metadata or alignment changed")
	}
	w, _ := f.Tensor("weight")
	matrix, err := NewMatrix(w, 1024)
	if err != nil {
		t.Fatal(err)
	}
	got, err := matrix.MatVec([]float32{1, 2, 3})
	if err != nil || got[0] != 14 || got[1] != 32 {
		t.Fatal(got, err)
	}
	if _, err = w.Bytes(1); err == nil {
		t.Fatal("ignored byte budget")
	}
	if _, err = Read(bytes.NewReader(buffer.Bytes()[:buffer.Len()-24]), int64(buffer.Len()-24)); err == nil {
		t.Fatal("accepted truncated weights")
	}
}
func TestQuantizationBlocks(t *testing.T) {
	for _, kind := range []Type{F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K} {
		t.Run(kind.String(), func(t *testing.T) {
			block, size, _ := layout(kind)
			data := make([]byte, int(size))
			want := make([]float32, int(block))
			switch kind {
			case F32:
				binary.LittleEndian.PutUint32(data, math.Float32bits(1.5))
				want[0] = 1.5
			case F16:
				binary.LittleEndian.PutUint16(data, 0x3e00)
				want[0] = 1.5
			case BF16:
				binary.LittleEndian.PutUint16(data, 0x3fc0)
				want[0] = 1.5
			case Q8_0:
				binary.LittleEndian.PutUint16(data, 0x3800)
				for i := range want {
					value := i - 16
					data[2+i] = byte(int8(value))
					want[i] = float32(value) * 0.5
				}
			case Q4_0, Q4_1, Q5_0, Q5_1:
				binary.LittleEndian.PutUint16(data, 0x3800)
				offset := 2
				minimum := float32(0)
				bits := 4
				if kind == Q4_1 || kind == Q5_1 {
					binary.LittleEndian.PutUint16(data[2:], 0xbc00)
					offset = 4
					minimum = -1
				}
				if kind == Q5_0 || kind == Q5_1 {
					bits = 5
					offset += 4
				}
				var high uint32
				for i := range want {
					q := i % (1 << bits)
					data[offset+i%16] |= byte(q&15) << uint((i/16)*4)
					if q&16 != 0 {
						high |= 1 << uint(i)
					}
					value := q
					if kind == Q4_0 {
						value -= 8
					}
					if kind == Q5_0 {
						value -= 16
					}
					want[i] = 0.5*float32(value) + minimum
				}
				if bits == 5 {
					binary.LittleEndian.PutUint32(data[offset-4:], high)
				}
			case Q4_K:
				binary.LittleEndian.PutUint16(data, 0x3800)
				binary.LittleEndian.PutUint16(data[2:], 0x3000)
				for g := 0; g < 8; g++ {
					scale, minimum := byte(7+g*7), byte(63-g*5)
					if g < 4 {
						data[4+g] |= scale
						data[8+g] |= minimum
					} else {
						data[8+g] = (scale & 15) | ((minimum & 15) << 4)
						data[g] |= (scale >> 4) << 6
						data[4+g] |= (minimum >> 4) << 6
					}
					for j := 0; j < 32; j++ {
						q := byte((g*3 + j) % 16)
						data[16+(g/2)*32+j] |= q << uint((g%2)*4)
						want[g*32+j] = 0.5*float32(scale)*float32(q) - 0.125*float32(minimum)
					}
				}
			case Q6_K:
				binary.LittleEndian.PutUint16(data[208:], 0x3400)
				for g := 0; g < 16; g++ {
					scale := g - 8
					data[192+g] = byte(int8(scale))
				}
				for i := range want {
					part, local := i/128, i%128
					group, lane := local/32, local%32
					q := byte((i * 13) % 64)
					data[part*64+(group%2)*32+lane] |= (q & 15) << uint((group/2)*4)
					data[128+part*32+lane] |= (q >> 4) << uint(group*2)
					want[i] = 0.25 * float32(i/16-8) * float32(int(q)-32)
				}
			}
			got, err := Decode(kind, data, int(block))
			if err != nil {
				t.Fatal(err)
			}
			for i, v := range got {
				if v != want[i] {
					t.Fatalf("element %d: %g want %g", i, v, want[i])
				}
			}
			matrix, err := NewPackedMatrix(1, int(block), kind, data)
			if err != nil {
				t.Fatal(err)
			}
			input := make([]float32, int(block))
			sum := float64(0)
			for i := range input {
				input[i] = float32(i%7 - 3)
				sum += float64(input[i]) * float64(want[i])
			}
			product, err := matrix.MatVec(input)
			if err != nil || product[0] != float32(sum) {
				t.Fatal(product, err)
			}
		})
	}
	if half(1) != float32(math.Ldexp(1, -24)) || !math.IsInf(float64(half(0x7c00)), 1) {
		t.Fatal("half conversion")
	}
}
func FuzzRead(f *testing.F) {
	f.Add([]byte("GGUF"))
	var buffer bytes.Buffer
	_ = Write(&buffer, map[string]any{"name": "test"}, nil)
	f.Add(buffer.Bytes())
	f.Fuzz(func(t *testing.T, data []byte) { _, _ = Read(bytes.NewReader(data), int64(len(data))) })
}
