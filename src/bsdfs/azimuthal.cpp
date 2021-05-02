/*
    This file is part of Mitsuba, a physically based rendering system.

    Copyright (c) 2007-2014 by Wenzel Jakob and others.

    Mitsuba is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License Version 3
    as published by the Free Software Foundation.

    Mitsuba is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include "math.h"
#include "InterpolatedDistribution1D.hpp"

MTS_NAMESPACE_BEGIN

class Azimuthal {
public:
    static const int AzimuthalResolution = 64;

    void sample(float cosThetaD, float xi, float &phi, float &pdf) const
    {
        float v = (AzimuthalResolution - 1)*cosThetaD;

        int x;
        _sampler->warp(v, xi, x);

        phi = 2.0f*M_PI*(x + xi)*(1.0f/AzimuthalResolution);
        pdf = _sampler->pdf(v, x)*float(AzimuthalResolution*(1.0f / (2.0f * M_PI)));
    }

    Vector3f eval(float phi, float cosThetaD) const
    {
        float u = (AzimuthalResolution - 1)*phi*(1.0f / (2.0f * M_PI));
        float v = (AzimuthalResolution - 1)*cosThetaD;
        int x0 = math::clamp(int(u), 0, AzimuthalResolution - 2);
        int y0 = math::clamp(int(v), 0, AzimuthalResolution - 2);
        int x1 = x0 + 1;
        int y1 = y0 + 1;
        u = math::clamp(u - x0, 0.0f, 1.0f);
        v = math::clamp(v - y0, 0.0f, 1.0f);

        return (_table[x0 + y0*AzimuthalResolution]*(1.0f - u) + _table[x1 + y0*AzimuthalResolution]*u)*(1.0f - v) +
               (_table[x0 + y1*AzimuthalResolution]*(1.0f - u) + _table[x1 + y1*AzimuthalResolution]*u)*v;
    }

    float pdf(float phi, float cosThetaD) const
    {
        float u = (AzimuthalResolution - 1)*phi*(1.0f / (2.0f * M_PI));
        float v = (AzimuthalResolution - 1)*cosThetaD;
        return _sampler->pdf(v, int(u))*float(AzimuthalResolution*(1.0f / (2.0f * M_PI)));
    }

    float weight(float cosThetaD) const
    {
        float v = (AzimuthalResolution - 1)*cosThetaD;
        return _sampler->sum(v)*(2.0f * M_PI/AzimuthalResolution);
    }
private:
    std::unique_ptr<Vector3f[]> _table;
    std::unique_ptr<InterpolatedDistribution1D> _sampler;
};