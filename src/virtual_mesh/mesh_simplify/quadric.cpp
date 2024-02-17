#include <quadric.h>
#include <luisa/core/mathematics.h>
#include <luisa/core/logging.h>
#include <luisa/core/basic_types.h>

// namespace virtual_mesh {

using namespace luisa;
// using namespace luisa::compute;
using namespace fmt;

double Quadric::eval(double3 p) {
    double3x3 m = {
        {xx, xy, xz},
        {xy, yy, yz},
        {xz, yz, zz}
    };
    return dot(p, m * p) + 2 * dot(p, double3{dx, dy, dz}) + d2;
}

void Quadric::from_edge(double3 p0, double3 p1, double3 p2) {
    double3 fn = normalize(cross(p1 - p0, p2 - p0));
    double3 p01 = p1 - p0;

    double3 n = cross(p01, fn);
    double d = -dot(n, p0);

    a = length(p01);
    // a = 1;

    xx = n.x * n.x * a;
    yy = n.y * n.y * a;
    zz = n.z * n.z * a;

    xy = n.x * n.y * a;
    xz = n.x * n.z * a;
    yz = n.y * n.z * a;

    dx = d * n.x * a;
    dy = d * n.y * a;
    dz = d * n.z * a;
    d2 = d * d * a;
}

void QuadricAttr::zero(int num_attr) {
    m = {0};
    for (int i = 0; i < num_attr; i++) {
        gs[i] = {0};
    }
}

void QuadricAttr::plane_info(
    luisa::double3 p0, luisa::double3 p1, luisa::double3 p2,
    double* attr0, double* attr1, double* attr2,
    int num_attr,
    luisa::double4 &n, luisa::double4 *g
) {
    double3 p01 = p1 - p0;
    double3 p02 = p2 - p0;

    double3 tn = normalize(cross(p01, p02));
    double d = -dot(tn, p0);

    n = make_double4(tn, d);

    // 求解顶点属性梯度g
    // dot(p01, g) = a1 - a0
    // dot(p02, g) = a2 - a0
    // dot(n, g) = 0
    double3x3 gm = {
        {p01.x, p02.x, n.x},
        {p01.y, p02.y, n.y},
        {p01.z, p02.z, n.z}
    };
    double3x3 inv_gm;
    bool binv = inverse(gm, inv_gm);

    for (int i = 0; i < num_attr; i++) {
        double a0 = attr0[i];
        double a1 = attr1[i];
        double a2 = attr2[i];

        double3 tg;

        if (!binv) {
            tg = double3{0};
        }
        else {
            double3 b = {a1 - a0, a2 - a0, 0};
            tg = inv_gm * b;
        }

        double d = a0 - dot(tg, p0);

        g[i] = {tg.x, tg.y, tg.z, d};
    }
}

void QuadricAttr::from_plane(
    double3 p0, double3 p1, double3 p2,
    double* attr0, double* attr1, double* attr2,
    int num_attr
) {
    double3 p01 = p1 - p0;
    double3 p02 = p2 - p0;

    double3 n = cross(p01, p02);
    double L = length(n);
    double area = 0.5 * L;

    n /= L;

    if (area < 1e-12) {
        zero(num_attr);
        return;
    }

    double d = -dot(n, p0);
    
    m.xx = n.x * n.x;
    m.yy = n.y * n.y;
    m.zz = n.z * n.z;

    m.xy = n.x * n.y;
    m.xz = n.x * n.z;
    m.yz = n.y * n.z;

    m.dx = d * n.x;
    m.dy = d * n.y;
    m.dz = d * n.z;
    m.d2 = d * d;

    // 求解顶点属性梯度g
    // dot(p01, g) = a1 - a0
    // dot(p02, g) = a2 - a0
    // dot(n, g) = 0
    double3x3 gm = {
        {p01.x, p02.x, n.x},
        {p01.y, p02.y, n.y},
        {p01.z, p02.z, n.z}
    };
    double3x3 inv_gm;
    bool binv = inverse(gm, inv_gm);

    for (int i = 0; i < num_attr; i++) {
        double a0 = attr0[i];
        double a1 = attr1[i];
        double a2 = attr2[i];

        double3 g;

        if (!binv) {
            g = double3{0};
        }
        else {
            double3 b = {a1 - a0, a2 - a0, 0};
            g = inv_gm * b;
        }

        double d = a0 - dot(g, p0);

        gs[i] = {g.x, g.y, g.z, d};

        m.xx += g.x * g.x;
        m.yy += g.y * g.y;
        m.zz += g.z * g.z;

        m.xy += g.x * g.y;
        m.xz += g.x * g.z;
        m.yz += g.y * g.z;

        m.dx += d * g.x;
        m.dy += d * g.y;
        m.dz += d * g.z;
        m.d2 += d * d;
    }

    // m.a = 1;

    m.xx *= area;
    m.yy *= area;
    m.zz *= area;

    m.xy *= area;
    m.xz *= area;
    m.yz *= area;

    m.dx *= area;
    m.dy *= area;
    m.dz *= area;
    m.d2 *= area;

    m.a = area;

    for (int i = 0; i < num_attr; i++) {
        gs[i].gx *= area;
        gs[i].gy *= area;
        gs[i].gz *= area;
        gs[i].d *= area;
    }
}

void QuadricAttr::add(const QuadricAttr &b, int num_attr) {
    m += b.m;
    for (int i = 0; i < num_attr; i++) {
        gs[i] += b.gs[i];
    }
}

double QuadricAttr::calc_attr_with_error(double3 opt_p, float* attr, int num_attr) {
    if (m.a < 1e-12) {
        for (int i = 0; i < num_attr; i++) attr[i] = 0;
        return 0;
    }

    double3 p = {opt_p.x, opt_p.y, opt_p.z};
    double3x3 tm = {
        {m.xx, m.xy, m.xz},
        {m.xy, m.yy, m.yz},
        {m.xz, m.yz, m.zz}
    };
    double error = dot(p, tm * p) + 2 * dot(p, double3{m.dx, m.dy, m.dz}) + m.d2;

    for (int i = 0; i < num_attr; i++) {
        QuadricGrad tg = gs[i];
        double3 g = {tg.gx, tg.gy, tg.gz};
        double t = dot(p, g) + tg.d;
        double s = t / m.a;

        attr[i] = s;
        // error += m.a * s * s - 2 * dot(p, g) * s - 2 * tg.d * s;
        error -= t * s;
    }

    if (error < 0 || isnan(error)) {
        error = 0;
    }
    return error;
}

QuadricOptimizer::QuadricOptimizer(int num_wedge, int num_attr, luisa::vector<luisa::ubyte> &wedge_quadrics)
    :num_wedge(num_wedge), num_attr(num_attr), wedge_quadrics(wedge_quadrics)
{
    m = {};
    bb = {};
    dn = {};
    bd = {};
    // a = 0;

    for (int i = 0; i < num_wedge; i++) {
        QuadricAttr& q = QuadricAttr::get(wedge_quadrics, num_attr, i);
        add_wedge_quadric(q);
    }
}

void QuadricOptimizer::add_wedge_quadric(QuadricAttr& q) {
    if (q.m.a < 1e-12) return;

    m.xx += q.m.xx;
    m.yy += q.m.yy;
    m.zz += q.m.zz;
    
    m.xy += q.m.xy;
    m.xz += q.m.xz;
    m.yz += q.m.yz;

    dn += double3{q.m.dx, q.m.dy, q.m.dz};
    // a += q.m.a;

    double inv_a = 1 / q.m.a;

    for (int j = 0; j < num_attr; j++) {
        QuadricGrad g = q.gs[j];

        bb.xx += g.gx * g.gx * inv_a;
        bb.yy += g.gy * g.gy * inv_a;
        bb.zz += g.gz * g.gz * inv_a;

        bb.xy += g.gx * g.gy * inv_a;
        bb.xz += g.gx * g.gz * inv_a;
        bb.yz += g.gy * g.gz * inv_a;

        bd += double3{g.gx, g.gy, g.gz} * (g.d * inv_a);
    }
}

void QuadricOptimizer::add_edge_quadric(Quadric q) {
    m.xx += q.xx;
    m.yy += q.yy;
    m.zz += q.zz;
    
    m.xy += q.xy;
    m.xz += q.xz;
    m.yz += q.yz;

    dn += double3{q.dx, q.dy, q.dz};
}

bool QuadricOptimizer::optimize(luisa::double3 &p) {
    // if (a < 1e-12) {
    //     return false;
    // }

    // double inv_a = 1 / a;

    // M = C = 1/a * B*Bt
    SymMat M;
    M.xx = m.xx - bb.xx;
    M.yy = m.yy - bb.yy;
    M.zz = m.zz - bb.zz;

    M.xy = m.xy - bb.xy;
    M.xz = m.xz - bb.xz;
    M.yz = m.yz - bb.yz;

    double3 b = bd - dn;

    // 解线性方程求导数为0点
    double3x3 mat = {
        {M.xx, M.xy, M.xz},
        {M.xy, M.yy, M.yz},
        {M.xz, M.yz, M.zz}
    };
    double3x3 inv_m;
    bool binv = inverse(mat, inv_m);

    if (!binv) {
        return false;
    }

    p = inv_m * b;

    return true;
}

double QuadricOptimizer::calc_attr_with_error(double3 opt_p, float* attr) {
    double error = 0;
    for (int i = 0; i < num_wedge; i++) {
        QuadricAttr& q = QuadricAttr::get(wedge_quadrics, num_attr, i);
        error += q.calc_attr_with_error(opt_p, attr + i * num_attr, num_attr);
    }
    return error;
}

// }
