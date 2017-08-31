package blasfuzz

import (
	"encoding/binary"
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
)

var (
	decode = binary.LittleEndian
)

type FuzzData struct {
	Data   []byte
	Failed bool
}

// Bools extracts a byte's worth of booleans
func (fd *FuzzData) Bools() [8]bool {
	bools := [8]bool{}
	if len(fd.Data) < 1 {
		fd.Failed = true
		return bools
	}
	b := fd.Data[0]
	for i := range bools {
		bools[i] = b&(1<<uint(i)) != 0
	}
	return bools
}

// GetInt gets an integer with the given number of bytes
func (fd *FuzzData) Int(b int) (n int) {
	if len(fd.Data) < b {
		fd.Failed = true
		return 0
	}
	switch b {
	case 1:
		return int(fd.Data[0])
	case 2:
		return int(decode.Uint16(fd.Data[:2:2]))
	default:
		panic("not coded")
	}
	fd.Data = fd.Data[b:]
}

func (fd *FuzzData) IntS(ln, b int) []int {
	x := make([]int, ln)
	for i := range x {
		x[i] = fd.Int(b)
	}
	return x
}

func (fd *FuzzData) F64() float64 {
	if len(fd.Data) < 8 {
		fd.Failed = true
		return math.NaN()
	}
	uint64 := decode.Uint64(fd.Data[:8:8])
	fd.Data = fd.Data[8:]
	float64 := math.Float64frombits(uint64)
	return float64
}

func (fd *FuzzData) F64S(ln int) []float64 {
	x := make([]float64, ln)
	for i := range x {
		x[i] = fd.F64()
	}
	return x
}

// Panics returns the error if panics
func CatchPanic(f func()) (err interface{}) {
	defer func() {
		err = recover()
	}()
	f()
	return
}

// SameError checks that the two errors are the same if either of them are non-nil.
func SamePanic(str string, c, native interface{}) {
	if c != nil && native == nil {
		panic(fmt.Sprintf("Case %s: c panics, native doesn't: %v", str, c))
	}
	if c == nil && native != nil {
		panic(fmt.Sprintf("Case %s: native panics, c doesn't: %v", str, native))
	}
	if c != native {
		panic(fmt.Sprintf("Case %s: Error mismatch.\nC is: %v\nNative is: %v", str, c, native))
	}
}

func CloneF64S(x []float64) []float64 {
	n := make([]float64, len(x))
	copy(n, x)
	return n
}

func SameInt(str string, c, native int) {
	if c != native {
		panic(fmt.Sprintf("Case %s: Int mismatch. c = %v, native = %v.", str, c, native))
	}
}

func SameF64Approx(str string, c, native, absTol, relTol float64) {
	if math.IsNaN(c) && math.IsNaN(native) {
		return
	}
	if !floats.EqualWithinAbsOrRel(c, native, absTol, relTol) {
		cb := math.Float64bits(c)
		nb := math.Float64bits(native)
		same := floats.EqualWithinAbsOrRel(c, native, absTol, relTol)
		panic(fmt.Sprintf("Case %s: Float64 mismatch. c = %v, native = %v\n cb: %v, nb: %v\n%v,%v,%v", str, c, native, cb, nb, same, absTol, relTol))
	}
}

func SameF64S(str string, c, native []float64) {
	if !floats.Same(c, native) {
		panic(fmt.Sprintf("Case %s: []float64 mismatch. c = %v, native = %v.", str, c, native))
	}
}

func SameF64SApprox(str string, c, native []float64, absTol, relTol float64) {
	if len(c) != len(native) {
		panic(str)
	}
	for i, v := range c {
		SameF64Approx(str, v, native[i], absTol, relTol)
	}
}
