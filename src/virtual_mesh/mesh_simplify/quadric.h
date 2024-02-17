#pragma once

#include <luisa/core/basic_types.h>
#include <luisa/core/stl/vector.h>
#include <float3.h>
// #include <double3.h>
// #include <assert.h>

// namespace virtual_mesh {

struct Quadric {
	double xx, yy, zz;
	double xy, xz, yz;
	double dx, dy, dz;
	double d2;
	
	double a;

	// 以 p0,p1,p2 为三角形，以 p0,p1 为边的 edge quadric
	void from_edge(luisa::double3 p0, luisa::double3 p1, luisa::double3 p2);

	double eval(luisa::double3 p);
};

struct QuadricGrad {
    double gx, gy, gz;
	double d;
};

// 为了支持可变长度并且存储在线性内存（QuadricAttr每个面都需要一个，避免大量vector创建）
// 需要开辟一个字节数组并强制转换为QuadricAttr使用，g[]即跟随在m后的数组
struct QuadricAttr {
	Quadric m;
	QuadricGrad gs[];

	static void plane_info(
		luisa::double3 p0, luisa::double3 p1, luisa::double3 p2,
		double* attr0, double* attr1, double* attr2,
		int num_attr,
		luisa::double4 &n, luisa::double4 *g
	);

	void from_plane(
		luisa::double3 p0, luisa::double3 p1, luisa::double3 p2,
		double* attr0, double* attr1, double* attr2,
		int num_attr
	);
	void zero(int num_attr);
	void add(const QuadricAttr &b, int num_attr);
	double calc_attr_with_error(luisa::double3 opt_p, float* attr, int num_attr);

	static QuadricAttr& get(luisa::vector<luisa::ubyte> &quadrics, int num_attr, int idx) {
		const int quadric_size = sizeof(Quadric) + sizeof(QuadricGrad) * num_attr;
		return *(QuadricAttr*)(&quadrics[idx * quadric_size]);
	}
};

struct QuadricOptimizer {
	int num_wedge;
	int num_attr;
	luisa::vector<luisa::ubyte> &wedge_quadrics;

	// 对称矩阵
    struct SymMat {
        double xx, yy, zz;
        double xy, xz, yz;
    };

	// wedge quadric matrix
    SymMat m, bb;
    luisa::double3 dn, bd;

	// 将不同wedge的面积相加是错的
    // double a;

	QuadricOptimizer(int num_wedge, int num_attr, luisa::vector<luisa::ubyte> &wedge_quadrics);

	void add_wedge_quadric(QuadricAttr& q);
	void add_edge_quadric(Quadric q);

	bool optimize(luisa::double3 &p);
	double calc_attr_with_error(luisa::double3 opt_p, float* attr);
};

inline Quadric operator+(const Quadric &a, const Quadric &b) {
	Quadric c;
	c.xx = a.xx + b.xx;
	c.yy = a.yy + b.yy;
	c.zz = a.zz + b.zz;

	c.xy = a.xy + b.xy;
	c.xz = a.xz + b.xz;
	c.yz = a.yz + b.yz;

	c.dx = a.dx + b.dx;
	c.dy = a.dy + b.dy;
	c.dz = a.dz + b.dz;
	c.d2 = a.d2 + b.d2;

	c.a = a.a + b.a;

	return c;
}
inline void operator+=(Quadric &a, const Quadric &b) {
	a = a + b;
}

inline QuadricGrad operator+(const QuadricGrad &a, const QuadricGrad &b) {
	QuadricGrad c;
	c.gx = a.gx + b.gx;
	c.gy = a.gy + b.gy;
	c.gz = a.gz + b.gz;

	c.d = a.d + b.d;
	return c;
}
inline void operator+=(QuadricGrad &a, const QuadricGrad &b) {
	a = a + b;
}

// }