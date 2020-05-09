//
// Created by Hugues Thomas.
//

#ifndef SPHERICAL_GRAPH_POINTS_H
#define SPHERICAL_GRAPH_POINTS_H

#include <vector>
#include <unordered_map>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>

#include <time.h>

class PointXYZ {
public:
    float x, y, z;
    // Methods
    PointXYZ() {x = 0, y = 0; z = 0;}
    PointXYZ( float x0, float y0, float z0) { x=x0; y=y0; z=z0;}

    // array type accessor
    float operator [] (int i) const {
        if (i==0) return x;
        else if(i == 1) return y;
        else return z;
    }

    // operations
    float dot(const PointXYZ P) const { // dot product
        return P.x * x + P.y * y + P.z * z;
    }

    float sq_norm() const{ // square norm
        return x*x + y*y + z*z;
    }

    float l2_norm() const{
        return std::sqrt(sq_norm());
    }

    float l1_norm() const{
        return std::abs(x) + std::abs(y) + std::abs(z);
    }

    float l_inf_norm() const{
        return std::max(std::max(std::abs(x), std::abs(y)), std::abs(z));
    }

    PointXYZ cross(const PointXYZ P) { // cross product
        return PointXYZ(y * P.z - z*P.y, z*P.x - x*P.z, x*P.y - y*P.x);
    }

    PointXYZ& operator +=(const PointXYZ& P)
    {
        x += P.x;
        y += P.y;
        z += P.z;
        return *this;
    }

    PointXYZ& operator -=(const PointXYZ& P){
        x -= P.x;
        y -= P.y;
        z -= P.z;
        return *this;
    }

    PointXYZ& operator *=(const PointXYZ& P){
        x *= P.x;
        y *= P.y;
        z *= P.z;
        return *this;
    }


};

// Point Opperations
// *****************

inline PointXYZ operator + (const PointXYZ A, const PointXYZ B)
{
    return PointXYZ(A.x + B.x, A.y + B.y, A.z + B.z);
}

inline PointXYZ operator - (const PointXYZ A, const PointXYZ B)
{
    return PointXYZ(A.x - B.x, A.y - B.y, A.z - B.z);
}

inline PointXYZ operator * (const PointXYZ P, const float a)
{
    return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline PointXYZ operator * (const float a, const PointXYZ P)
{
    return PointXYZ(P.x * a, P.y * a, P.z * a);
}

inline std::ostream& operator << (std::ostream& os, const PointXYZ P)
{
    return os << "[" << P.x << ", " << P.y << ", " << P.z << "]";
}

inline bool operator == (const PointXYZ A, const PointXYZ B)
{
    return A.x == B.x && A.y == B.y && A.z == B.z;
}

inline PointXYZ floor(const PointXYZ P)
{
    return PointXYZ(std::floor(P.x), std::floor(P.y), std::floor(P.z));
}


PointXYZ max_point(std::vector<PointXYZ> points)
{
    // Initiate limits
    PointXYZ maxP(points[0]);

    // Loop over all points
    for (auto p : points)
    {
        if (p.x > maxP.x)
            maxP.x = p.x;

        if (p.y > maxP.y)
            maxP.y = p.y;

        if (p.z > maxP.z)
            maxP.z = p.z;
    }

    return maxP;
}
PointXYZ min_point(std::vector<PointXYZ> points)
{
    // Initiate limits
    PointXYZ minP(points[0]);

    // Loop over all points
    for (auto p : points)
    {
        if (p.x < minP.x)
            minP.x = p.x;

        if (p.y < minP.y)
            minP.y = p.y;

        if (p.z < minP.z)
            minP.z = p.z;
    }

    return minP;
}

#endif //SPHERICAL_GRAPH_POINTS_H
