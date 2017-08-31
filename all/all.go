// all fuzzes all of the BLAS functions
package all

import (
	"fmt"

	"github.com/btracey/blasfuzz"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/netlib/blas/netlib"
)

var (
	cImpl  = netlib.Implementation{}
	goImpl = gonum.Implementation{}
)

func Fuzz(data []byte) int {
	bfuzz := blasfuzz.FuzzData{Data: data}

	n := bfuzz.Int(1)

	// Construct slice 1
	incX := bfuzz.Int(1)
	lenX := bfuzz.Int(2)
	x := bfuzz.F64S(lenX)

	// Construct slice 2
	incY := bfuzz.Int(1)
	lenY := bfuzz.Int(2)
	y := bfuzz.F64S(lenY)

	// Construct matrix 1
	m1 := bfuzz.Int(1)
	n1 := bfuzz.Int(1)
	ld1 := bfuzz.Int(1)
	lenA := ld1*m1 + n1
	a := bfuzz.F64S(lenA)

	// Construct matrix 2
	m2 := bfuzz.Int(1)
	n2 := bfuzz.Int(1)
	ld2 := bfuzz.Int(1)
	lenB := ld2*m2 + n2
	b := bfuzz.F64S(lenB)

	// Construct matrix 3
	m3 := bfuzz.Int(1)
	n3 := bfuzz.Int(1)
	ld3 := bfuzz.Int(1)
	lenC := ld3*m3 + n3
	c := bfuzz.F64S(lenC)

	// Generate a couple of parameters and booleans
	nParams := 8
	params := bfuzz.F64S(nParams)

	// Generate a couple of booleans
	bools := bfuzz.Bools()

	// Generate an integer
	iParam := bfuzz.Int(1)

	_, _, _ = a, b, c
	_ = bools

	if bfuzz.Failed {
		return 0
	}

	// Test the functions
	level1Test(n, x, lenX, incX, y, lenY, incY, params, iParam)

	return 1
}

func level1Test(n int, x []float64, lenX, incX int, y []float64, lenY, incY int, params []float64, iParam int) {
	alpha := params[0]
	beta := params[1]

	flag := iParam
	if flag < 0 {
		flag = -flag
	}
	flag = flag%4 - 2

	drotm := blas.DrotmParams{
		Flag: blas.Flag(flag),
		H:    [4]float64{params[0], params[1], params[2], params[3]},
	}

	str1 := fmt.Sprintf("Case: N = %v, IncX = %v, x = %#v, alpha = %v", n, incX, x, alpha)
	str2 := fmt.Sprintf("Case: N = %v\n IncX: %v, x: %v\nIncY: %v, y: %v\n alpha: %v", n, incX, x, incY, y, alpha)
	str3 := fmt.Sprintf("Case: N = %v\n IncX: %v, x: %v\nIncY: %v, y: %v\n alpha: %v beta: %v", n, incX, x, incY, y, alpha, beta)
	str4 := fmt.Sprintf("Case: N = %v, IncX = %v, x = %#v, drotm = %v", n, incX, x, drotm)

	testDrot(str3, n, x, incX, y, incY, alpha, beta)
	testDrotm(str4, n, x, incX, y, incY, drotm)
	testDswap(str2, n, x, incX, y, incY)
	testDscal(str1, n, x, incX, alpha)
	testDcopy(str2, n, x, incX, y, incY)
	testDaxpy(str2, n, alpha, x, incX, y, incY)
	testDdot(str2, n, x, incX, y, incY)
	testDnrm2(str1, n, x, incX)
	testDasum(str1, n, x, incX)
	testIdamax(str1, n, x, incX)
}

func testIdamax(str string, n int, x []float64, incX int) {
	var natAns int
	nat := func() { natAns = goImpl.Idamax(n, x, incX) }
	errNative := blasfuzz.CatchPanic(nat)

	cx := blasfuzz.CloneF64S(x)
	var cAns int
	c := func() { cAns = cImpl.Idamax(n, cx, incX) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cx, x)
	// Known issue: If the slice contains NaN the answer may vary
	if !floats.HasNaN(x) {
		blasfuzz.SameInt(str, cAns, natAns)
	}
}

func testDnrm2(str string, n int, x []float64, incX int) {
	var natAns float64
	nat := func() { natAns = goImpl.Dnrm2(n, x, incX) }
	errNative := blasfuzz.CatchPanic(nat)

	cx := blasfuzz.CloneF64S(x)
	var cAns float64
	c := func() { cAns = cImpl.Dnrm2(n, cx, incX) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cx, x)
	blasfuzz.SameF64Approx(str, cAns, natAns, 1e-13, 1e-13)
}

func testDasum(str string, n int, x []float64, incX int) {
	var natAns float64
	nat := func() { natAns = goImpl.Dasum(n, x, incX) }
	errNative := blasfuzz.CatchPanic(nat)

	cx := blasfuzz.CloneF64S(x)
	var cAns float64
	c := func() { cAns = cImpl.Dasum(n, cx, incX) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cx, x)
	blasfuzz.SameF64Approx(str, cAns, natAns, 1e-13, 1e-13)
}

func testDscal(str string, n int, x []float64, incX int, alpha float64) {
	natX := blasfuzz.CloneF64S(x)
	nat := func() { goImpl.Dscal(n, alpha, natX, incX) }
	errNative := blasfuzz.CatchPanic(nat)

	cx := blasfuzz.CloneF64S(x)
	c := func() { cImpl.Dscal(n, alpha, cx, incX) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cx, natX)
}

func testDaxpy(str string, n int, alpha float64, x []float64, incX int, y []float64, incY int) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)

	nat := func() { goImpl.Daxpy(n, alpha, natX, incX, natY, incY) }
	errNative := blasfuzz.CatchPanic(nat)
	c := func() { cImpl.Daxpy(n, alpha, cX, incX, cY, incY) }
	errC := blasfuzz.CatchPanic(c)
	/*
		if n == 0 {
			str2 := fmt.Sprintf("cerr = %v, naterr = %v", errC, errNative)
			panic(str2)
		}
	*/
	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
}

func testDcopy(str string, n int, x []float64, incX int, y []float64, incY int) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)

	nat := func() { goImpl.Dcopy(n, natX, incX, natY, incY) }
	errNative := blasfuzz.CatchPanic(nat)
	c := func() { cImpl.Dcopy(n, cX, incX, cY, incY) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
}

func testDswap(str string, n int, x []float64, incX int, y []float64, incY int) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)

	nat := func() { goImpl.Dswap(n, natX, incX, natY, incY) }
	errNative := blasfuzz.CatchPanic(nat)
	c := func() { cImpl.Dswap(n, cX, incX, cY, incY) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
}

func testDdot(str string, n int, x []float64, incX int, y []float64, incY int) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)

	var natAns float64
	nat := func() { natAns = goImpl.Ddot(n, natX, incX, natY, incY) }
	errNative := blasfuzz.CatchPanic(nat)
	var cAns float64
	c := func() { cAns = cImpl.Ddot(n, cX, incX, cY, incY) }
	errC := blasfuzz.CatchPanic(c)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
	blasfuzz.SameF64Approx(str, cAns, natAns, 1e-13, 1e-13)
}

func testDrot(str string, n int, x []float64, incX int, y []float64, incY int, c, s float64) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)

	nat := func() { goImpl.Drot(n, natX, incX, natY, incY, c, s) }
	errNative := blasfuzz.CatchPanic(nat)
	cFunc := func() { cImpl.Drot(n, cX, incX, cY, incY, c, s) }
	errC := blasfuzz.CatchPanic(cFunc)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
}

func testDrotm(str string, n int, x []float64, incX int, y []float64, incY int, param blas.DrotmParams) {
	natX := blasfuzz.CloneF64S(x)
	natY := blasfuzz.CloneF64S(y)
	cX := blasfuzz.CloneF64S(x)
	cY := blasfuzz.CloneF64S(y)
	nat := func() { goImpl.Drotm(n, natX, incX, natY, incY, param) }
	errNative := blasfuzz.CatchPanic(nat)
	cFunc := func() { cImpl.Drotm(n, cX, incX, cY, incY, param) }
	errC := blasfuzz.CatchPanic(cFunc)

	blasfuzz.SamePanic(str, errC, errNative)
	blasfuzz.SameF64S(str, cX, natX)
	blasfuzz.SameF64S(str, cY, natY)
}
