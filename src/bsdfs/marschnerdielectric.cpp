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

#include <mitsuba/render/bsdf.h>
#include <mitsuba/hw/basicshader.h>
#include "InterpolatedDistribution1D.hpp"
#include <mitsuba/core/warp.h>
#include "ior.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{thindielectric}{Thin dielectric material}
 * \order{4}
 * \parameters{
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{bk7} / 1.5046}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{air} / 1.000277}}
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 *     \parameter{specular\showbreak Transmittance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular transmission component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 * }
 *
 * This plugin models a \emph{thin} dielectric material that is embedded inside another
 * dielectric---for instance, glass surrounded by air. The interior of the material
 * is assumed to be so thin that its effect on transmitted rays is negligible,
 * Hence, light exits such a material without any form of angular deflection
 * (though there is still specular reflection).
 *
 * This model should be used for things like glass windows that were modeled using only a
 * single sheet of triangles or quads. On the other hand, when the window consists of
 * proper closed geometry, \pluginref{dielectric} is the right choice. This is illustrated below:
 *
 * \begin{figure}[h]
 * \setcounter{subfigure}{0}
 * \centering
 * \hfill
 * \subfloat[The \pluginref{dielectric} plugin models a single transition from one index of refraction to another]
 *     {\includegraphics[width=4.5cm]{images/bsdf_dielectric_figure.pdf}}\hfill
 * \subfloat[The \pluginref{thindielectric} plugin models a pair of interfaces causing a transient index of refraction change]
 *      {\includegraphics[width=4.5cm]{images/bsdf_thindielectric_figure.pdf}}\hfill
 * \subfloat[Windows modeled using a single sheet of geometry are the most frequent application of this BSDF]
 *      {\fbox{\includegraphics[width=4.5cm]{images/bsdf_thindielectric_window.jpg}}}\hspace*\fill
 * \caption{
 *     \label{fig:thindielectric-diff}
 *     An illustration of the difference between the \pluginref{dielectric} and \pluginref{thindielectric} plugins}
 * \end{figure}
 *
 * The implementation correctly accounts for multiple internal reflections
 * inside the thin dielectric at \emph{no significant extra cost}, i.e. paths
 * of the type $R, TRT, TR^3T, ..$ for reflection and $TT, TR^2, TR^4T, ..$ for
 * refraction, where $T$ and $R$ denote individual reflection and refraction
 * events, respectively.
 */
class Azimuthal {
public:
    Azimuthal(std::unique_ptr<Vector3f[]> table) : _table(std::move(table))
    {
        const int Size = AzimuthalResolution;

        std::vector<float> weights(Size*Size);
        for (int i = 0; i < Size*Size; ++i)
            weights[i] = _table[i].max();

        // Dilate weights slightly to stay conservative
        for (int y = 0; y < Size; ++y) {
            for (int x = 0; x < Size - 1; ++x)
                weights[x + y*Size] = std::max(weights[x + y*Size], weights[x + 1 + y*Size]);
            for (int x = Size - 1; x > 0; --x)
                weights[x + y*Size] = std::max(weights[x + y*Size], weights[x - 1 + y*Size]);
        }
        for (int x = 0; x < Size; ++x) {
            for (int y = 0; y < Size - 1; ++y)
                weights[x + y*Size] = std::max(weights[x + y*Size], weights[x + (y + 1)*Size]);
            for (int y = Size - 1; y > 0; --y)
                weights[x + y*Size] = std::max(weights[x + y*Size], weights[x + (y - 1)*Size]);
        }

        _sampler.reset(new InterpolatedDistribution1D(std::move(weights), Size, Size));
    }

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

class MarschnerDielectric : public BSDF {
public:
    MarschnerDielectric(const Properties &props) : BSDF(props) {
        /* Specifies the internal index of refraction at the interface */
        Float intIOR = lookupIOR(props, "intIOR", "benzene");

        /* Specifies the external index of refraction at the interface */
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive!");

        m_eta = intIOR / extIOR;

        m_diffuseReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("diffuseReflectance", Spectrum(0.5f)));
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(0.1f)));
        m_specularTransmittance = new ConstantSpectrumTexture(
            props.getSpectrum("specularTransmittance", Spectrum(0.1f)));
        m_exponent = new ConstantFloatTexture(
            props.getFloat("exponent", 30.0f));
        m_specularSamplingWeight = 0.0f;
    }

    MarschnerDielectric(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_eta = stream->readFloat();
        m_diffuseReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
        m_exponent = static_cast<Texture *>(manager->getInstance(stream));
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeFloat(m_eta);
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_specularTransmittance.get());
    }

    void configure() {
        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        m_specularTransmittance = ensureEnergyConservation(
            m_specularTransmittance, "specularTransmittance", 1.0f);

        m_components.clear();
        m_components.push_back(EDeltaReflection | EFrontSide 
            | ((!m_specularReflectance->isConstant()
              || !m_exponent->isConstant()) ? ESpatiallyVarying : 0));
        m_components.push_back(ENull | EFrontSide 
            | (m_specularTransmittance->isConstant() ? 0 : ESpatiallyVarying));
        m_components.push_back(EDiffuseReflection | EFrontSide
            | (m_diffuseReflectance->isConstant() ? 0 : ESpatiallyVarying));

        m_usesRayDifferentials = false;

         Float dAvg = m_diffuseReflectance->getAverage().getLuminance(),
              sAvg = m_specularReflectance->getAverage().getLuminance(),
              tAvg = m_specularTransmittance->getAverage().getLuminance();
        m_specularSamplingWeight = (sAvg + tAvg) / (dAvg + sAvg + tAvg);


        m_usesRayDifferentials =
            m_specularReflectance->usesRayDifferentials() ||
            m_specularTransmittance->usesRayDifferentials();

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else if (name == "specularTransmittance")
                m_specularTransmittance = static_cast<Texture *>(child);
            else if (name == "diffuseReflectance")
                m_diffuseReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }

    /// Transmission in local coordinates
    inline Vector transmit(const Vector &wi) const {
        return -wi;
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 2);

        if (Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
            return Spectrum(0.0f);

        // Spectrum result(0.0f);
        // if (sampleReflection) {
        //     Float alpha    = dot(bRec.wo, reflect(bRec.wi)),
        //           exponent = m_exponent->eval(bRec.its).average();

        //     if (alpha > 0.0f) {
        //         result += m_specularReflectance->eval(bRec.its) *
        //             ((exponent + 2) * INV_TWOPI * std::pow(alpha, exponent));
        //     }
        // }

        Spectrum result(0.0f);

        Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

        // Account for internal reflections: R' = R + TRT + TR^3T + ..
        if (R < 1)
            R += T*T * R / (1-R*R);

        if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
            if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);

            // result += m_specularReflectance->eval(bRec.its) * R;

            Float tl = std::abs(bRec.wi.x);
            Float te = std::abs(bRec.wo.x);
            Float sin_tl = std::sqrt(1-tl*tl);
            Float sin_te = std::sqrt(1-te*te);
            Float alpha = tl*te + sin_tl*sin_te;
            Float exponent = m_exponent->eval(bRec.its).average();
            if (alpha > 0.0f && bRec.wi.x * bRec.wo.x < 0) {

                // Add an empirical constant since the cone was making it too bright.
                Spectrum res = 0.15f * m_specularReflectance->eval(bRec.its) *
                    ((exponent + 2) * INV_FOURPI * std::pow(alpha, exponent));
                result += res;
            }
        } else {
            if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                return Spectrum(0.0f);

            result += m_specularTransmittance->eval(bRec.its) * (1 - R);
        }

        if (hasDiffuse)
            result += m_diffuseReflectance->eval(bRec.its) * INV_PI;

        return result * Frame::cosTheta(bRec.wo);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 2);

        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            (!sampleReflection && !sampleTransmission && !hasDiffuse))
            return 0.0f;

        Float diffuseProb = 0.0f, specProb = 0.0f;
        if (hasDiffuse)
            diffuseProb = warp::squareToCosineHemispherePdf(bRec.wo);

        if (sampleReflection) {
            Float alpha    = dot(bRec.wo, reflect(bRec.wi)),
                  exponent = m_exponent->eval(bRec.its).average();
            if (alpha > 0)
                specProb = std::pow(alpha, exponent) *
                    (exponent + 1.0f) / (2.0f * M_PI);
        }

        Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

        // Account for internal reflections: R' = R + TRT + TR^3T + ..
        if (R < 1)
            R += T*T * R / (1-R*R);

        Float result = 0.0f;
        Float tProb = 0.0f;
        Float rProb = 0.0f;

        if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
            // no reflection or transmission
            if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                if (hasDiffuse) {
                    return diffuseProb;
                } else {
                    return 0.0f;
                }

            // prob of transmission
            tProb = sampleTransmission ? R : 1.0f;
        } else {
            // no reflection or transmission
            if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
                if (hasDiffuse) {
                    return diffuseProb;
                } else {
                    return 0.0f;
                }

            // prob of reflection
            rProb = sampleReflection ? 1-R : 1.0f;
        }

        if (hasDiffuse && sampleReflection) {
            return m_specularSamplingWeight * specProb * rProb
                + (1-m_specularSamplingWeight) * diffuseProb;
        } else if (hasDiffuse && sampleTransmission) {
            return m_specularSamplingWeight * specProb * tProb
                + (1-m_specularSamplingWeight) * diffuseProb;
        } else if (hasDiffuse) {
            return diffuseProb;
        } else if (sampleReflection) {
            return rProb;
        } else if (sampleTransmission) {
            return tProb;
        } else {
            return 0.0f;
        }


        // if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
        //     if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         if (hasDiffuse) {
        //             return diffuseProb;
        //         } else {
        //             return 0.0f;
        //         }
        //     // has diffuse and has specular
        //     if (hasDiffuse) {
        //         // has transmittance 
        //         if (sampleTransmission) {
        //             specProb *= R;
        //         }
        //         return m_specularSamplingWeight * specProb + 
        //             m_diffuseSamplingWeight * diffuseProb; 
        //     } else {
        //         return sampleTransmission ? R : 1.0f;
        //     }
        // } else {
        //     if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         if (hasDiffuse) {
        //             return diffuseProb;
        //         } else {
        //             return 0.0f;
        //         }
        //     // has diffuse and has transmission
        //     if (hasDiffuse) {
        //         // has specular 
        //         if (sampleReflection) {
        //             specProb *= (1-R);
        //         }
        //         return m_transmittanceSamplingWeight * specProb + 
        //             m_diffuseReflectance * diffuseProb;
        //     } else {
        //         return sampleReflection ? 1-R : 1.0f;
        //     }
        // }
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
        bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1);
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 2);

        if (!sampleReflection && !sampleTransmission && !hasDiffuse)
            return Spectrum(0.0f);

        bool hasSpecular = sampleReflection || sampleTransmission;
        bool choseSpecular = hasSpecular;

        Point2 sample(_sample);

        if (hasDiffuse && hasSpecular) {
            if (sample.x <= m_specularSamplingWeight) {
                sample.x /= m_specularSamplingWeight;
            } else {
                sample.x = (sample.x - m_specularSamplingWeight)
                    / (1-m_specularSamplingWeight);
                choseSpecular = false;
            }
        }

        if (choseSpecular) {
            Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

            // Account for internal reflections: R' = R + TRT + TR^3T + ..
            if (R < 1)
                R += T*T * R / (1-R*R);

            if (sampleTransmission && sampleReflection) {
                if (sample.x <= R) {
                    bRec.sampledComponent = 0;
                    bRec.sampledType = EDeltaReflection;
                    bRec.wo = reflect(bRec.wi);
                    bRec.eta = 1.0f;
                    _pdf = R;

                    return m_specularReflectance->eval(bRec.its);
                } else {
                    bRec.sampledComponent = 1;
                    bRec.sampledType = ENull;
                    bRec.wo = transmit(bRec.wi);
                    bRec.eta = 1.0f;
                    _pdf = 1-R;

                    return m_specularTransmittance->eval(bRec.its);
                }
            } else if (sampleReflection) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDeltaReflection;
                bRec.wo = reflect(bRec.wi);
                bRec.eta = 1.0f;
                _pdf = 1.0f;

                return m_specularReflectance->eval(bRec.its) * R;
            } else if (sampleTransmission) {
                bRec.sampledComponent = 1;
                bRec.sampledType = ENull;
                bRec.wo = transmit(bRec.wi);
                bRec.eta = 1.0f;
                _pdf = 1.0f;

                return m_specularTransmittance->eval(bRec.its) * (1-R);
            }
        } else {
            bRec.wo = warp::squareToCosineHemisphere(sample);
            bRec.sampledComponent = 2;
            bRec.sampledType = EDiffuseReflection;
        }

        bRec.eta = 1.0f;

        _pdf = pdf(bRec, ESolidAngle);

        if (_pdf == 0)
            return Spectrum(0.0f);
        else
            return eval(bRec, ESolidAngle) / _pdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &_sample) const {
        bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool sampleTransmission = (bRec.typeMask & ENull)
                && (bRec.component == -1 || bRec.component == 1);
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 2);

        Point2 sample(_sample);

        if (!sampleReflection && !sampleTransmission && !hasDiffuse)
            return Spectrum(0.0f);

        bool hasSpecular = sampleReflection || sampleTransmission;
        bool choseSpecular = hasSpecular;

        if (hasDiffuse && hasSpecular) {
            if (sample.x <= m_specularSamplingWeight) {
                sample.x /= m_specularSamplingWeight;
            } else {
                sample.x = (sample.x - m_specularSamplingWeight)
                    / (1-m_specularSamplingWeight);
                choseSpecular = false;
            }
        }

        if (choseSpecular) {
            Float R = fresnelDielectricExt(Frame::cosTheta(bRec.wi), m_eta), T = 1-R;

            // Account for internal reflections: R' = R + TRT + TR^3T + ..
            if (R < 1)
                R += T*T * R / (1-R*R);

            if (sampleTransmission && sampleReflection) {
                if (sample.x <= R) {
                    bRec.sampledComponent = 0;
                    bRec.sampledType = EDeltaReflection;
                    bRec.wo = reflect(bRec.wi);
                    bRec.eta = 1.0f;

                    return m_specularReflectance->eval(bRec.its);
                } else {
                    bRec.sampledComponent = 1;
                    bRec.sampledType = ENull;
                    bRec.wo = transmit(bRec.wi);
                    bRec.eta = 1.0f;

                    return m_specularTransmittance->eval(bRec.its);
                }
            } else if (sampleReflection) {
                bRec.sampledComponent = 0;
                bRec.sampledType = EDeltaReflection;
                bRec.wo = reflect(bRec.wi);
                bRec.eta = 1.0f;

                return m_specularReflectance->eval(bRec.its) * R;
            } else if (sampleTransmission) {
                bRec.sampledComponent = 1;
                bRec.sampledType = ENull;
                bRec.wo = transmit(bRec.wi);
                bRec.eta = 1.0f;

                return m_specularTransmittance->eval(bRec.its) * (1-R);
            }
        } else {
            bRec.wo = warp::squareToCosineHemisphere(sample);
            bRec.sampledComponent = 2;
            bRec.sampledType = EDiffuseReflection;
        }

        bRec.eta = 1.0f;

        Float _pdf = pdf(bRec, ESolidAngle);

        if (_pdf == 0)
            return Spectrum(0.0f);
        else
            return eval(bRec, ESolidAngle) / _pdf;
    }

    Float getEta() const {
        /* The rrelative IOR across this interface is 1, since the internal
           material is thin: it begins and ends here. */
        return 1.0f;
    }

    Float getRoughness(const Intersection &its, int component) const {
        return 0.0f;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "MarschnerDielectric[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    Float m_eta;
    ref<Texture> m_specularTransmittance;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_diffuseReflectance;
    ref<Texture> m_exponent;
    Float m_specularSamplingWeight;
};

/* Fake glass shader -- it is really hopeless to visualize
   this material in the VPL renderer, so let's try to do at least
   something that suggests the presence of a transparent boundary */
class MarschnerDielectricShader : public Shader {
public:
    MarschnerDielectricShader(Renderer *renderer) :
        Shader(renderer, EBSDFShader) {
        m_flags = ETransparent;
    }

    Float getAlpha() const {
        return 0.3f;
    }

    void generateCode(std::ostringstream &oss,
            const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) < 0.0 || cosTheta(wo) < 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return vec3(inv_pi * cosTheta(wo));" << endl
            << "}" << endl
            << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    return " << evalName << "(uv, wi, wo);" << endl
            << "}" << endl;
    }


    MTS_DECLARE_CLASS()
};

Shader *MarschnerDielectric::createShader(Renderer *renderer) const {
    return new MarschnerDielectricShader(renderer);
}

MTS_IMPLEMENT_CLASS(MarschnerDielectricShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(MarschnerDielectric, false, BSDF)
MTS_EXPORT_PLUGIN(MarschnerDielectric, "Thin dielectric BSDF");
MTS_NAMESPACE_END
