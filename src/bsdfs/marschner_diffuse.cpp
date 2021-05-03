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
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/phase.h>
#include <mitsuba/render/medium.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/hw/basicshader.h>
#include <mitsuba/core/warp.h>
#include "../medium/materials.h"
#include "microfacet.h"
#include "rtrans.h"
#include "ior.h"
#include "math.h"
#include <iostream>
#include <array>
#include "gausssexylingerie.hpp"
#include "InterpolatedDistribution1D.hpp"

MTS_NAMESPACE_BEGIN

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

class Marschner : public BSDF {
public:
    Marschner(const Properties &props) : BSDF(props) {
        /* Specifies the internal index of refraction at the interface */
        Float intIOR = lookupIOR(props, "intIOR", "bk7");

        /* Specifies the external index of refraction at the interface */
        Float extIOR = lookupIOR(props, "extIOR", "air");

        if (intIOR < 0 || extIOR < 0)
            Log(EError, "The interior and exterior indices of "
                "refraction must be positive!");

        m_eta = intIOR / extIOR;
        _sigmaA = Vector3f(0.22);        

        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
        m_specularTransmittance = new ConstantSpectrumTexture(
            props.getSpectrum("specularTransmittance", Spectrum(1.0f)));
        
        m_diffuseReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("diffuseReflectance", Spectrum(0.5f)));

        m_nonlinear = props.getBoolean("nonlinear", false);

        MicrofacetDistribution distr(props);
        m_type = distr.getType();
        m_sampleVisible = distr.getSampleVisible();

        if (distr.isAnisotropic())
            Log(EError, "The 'roughplastic' plugin currently does not support "
                "anisotropic microfacet distributions!");

        m_alpha = new ConstantFloatTexture(distr.getAlpha());

        m_specularSamplingWeight = 0.0f;

        _roughness = 0.1f;
        // _betaR   = std::max(M_PI_FLT * 0.5f * _roughness, 0.04f);
        _betaR   = 0.1f;
        _betaTT  = _betaR*0.5f;
        _betaTRT = _betaR*2.0f;
        _scaleAngleRad = -0.1f;
        precomputeAzimuthalDistributions();
        _vR = _betaR * _betaR;
        _vTT = _betaTT * _betaTT;
        _vTRT = _betaTRT * _betaTRT;
    }

    Marschner(Stream *stream, InstanceManager *manager)
            : BSDF(stream, manager) {
        m_eta = stream->readFloat();
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularTransmittance = static_cast<Texture *>(manager->getInstance(stream));
        m_phase = static_cast<PhaseFunction *>(manager->getInstance(stream));

        m_type = (MicrofacetDistribution::EType) stream->readUInt();
        m_sampleVisible = stream->readBool();
        m_diffuseReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_alpha = static_cast<Texture *>(manager->getInstance(stream));
        m_nonlinear = stream->readBool();

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        stream->writeFloat(m_eta);
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_specularTransmittance.get());
        manager->serialize(stream, m_phase.get());

        stream->writeUInt((uint32_t) m_type);
        stream->writeBool(m_sampleVisible);
        manager->serialize(stream, m_diffuseReflectance.get());
        manager->serialize(stream, m_alpha.get());
        stream->writeBool(m_nonlinear);
    }

    void configure() {
        bool constAlpha = m_alpha->isConstant();
        if (m_phase == NULL)
            m_phase = static_cast<PhaseFunction *> (PluginManager::getInstance()->
                    createObject(MTS_CLASS(PhaseFunction), Properties("kkay")));

        /* Verify the input parameters and fix them if necessary */
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        m_specularTransmittance = ensureEnergyConservation(
            m_specularTransmittance, "specularTransmittance", 1.0f);

        m_components.clear();
        m_components.push_back(EDeltaReflection | EFrontSide | EBackSide
            | (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));
        m_components.push_back(ENull | EFrontSide | EBackSide
            | (m_specularTransmittance->isConstant() ? 0 : ESpatiallyVarying));
        m_components.push_back(EDiffuseReflection | EFrontSide
            | ((constAlpha && m_diffuseReflectance->isConstant())
                ? 0 : ESpatiallyVarying));

        Float dAvg = m_diffuseReflectance->getAverage().getLuminance(),
              sAvg = m_specularReflectance->getAverage().getLuminance();
        m_specularSamplingWeight = sAvg / (dAvg + sAvg);

        m_invEta2 = 1.0f / (m_eta*m_eta);

        if (!m_externalRoughTransmittance.get()) {
            /* Load precomputed data used to compute the rough
               transmittance through the dielectric interface */
            m_externalRoughTransmittance = new RoughTransmittance(m_type);

            m_externalRoughTransmittance->checkEta(m_eta);
            m_externalRoughTransmittance->checkAlpha(m_alpha->getMinimum().average());
            m_externalRoughTransmittance->checkAlpha(m_alpha->getMaximum().average());

            /* Reduce the rough transmittance data to a 2D slice */
            m_internalRoughTransmittance = m_externalRoughTransmittance->clone();
            m_externalRoughTransmittance->setEta(m_eta);
            m_internalRoughTransmittance->setEta(1/m_eta);

            /* If possible, even reduce it to a 1D slice */
            if (constAlpha)
                m_externalRoughTransmittance->setAlpha(
                    m_alpha->eval(Intersection()).average());
        }


        m_usesRayDifferentials =
            m_specularReflectance->usesRayDifferentials() ||
            m_specularTransmittance->usesRayDifferentials() ||
            m_alpha->usesRayDifferentials();

        BSDF::configure();
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(PhaseFunction))) {
            m_phase = static_cast<PhaseFunction *>(child);
        }
        else if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "alpha")
                m_alpha = static_cast<Texture *>(child);
            else if (name == "diffuseReflectance")
                m_diffuseReflectance = static_cast<Texture *>(child);
            else if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else if (name == "specularTransmittance")
                m_specularTransmittance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    /// Reflection in local coordinates
    inline Vector3f reflect(const Vector3f &wi) const {
        return Vector3f(-wi.x, -wi.y, wi.z);
    }

    /// Transmission in local coordinates
    inline Vector3f transmit(const Vector3f &wi) const {
        return -wi;
    }

    // Spectrum getDiffuseReflectance(const Intersection &its) const {
    //     return m_diffuseReflectance->eval(its);
    // }

    // Spectrum getSpecularReflectance(const Intersection &its) const {
    //     return m_specularReflectance->eval(its);
    // }

    static float I0(float x) {
        float result = 1.0f;
        float xSq = x*x;
        float xi = xSq;
        float denom = 4.0f;
        for (int i = 1; i <= 10; ++i) {
            result += xi/denom;
            xi *= xSq;
            denom *= 4.0f*float((i + 1)*(i + 1));
        }
        return result;
    }

    static float logI0(float x) {
        if (x > 12.0f)
            // More stable evaluation of log(I0(x))
            // See also https://publons.com/discussion/12/
            return x + 0.5f*(std::log(1.0f/(M_PI_FLT * 2.0f*x)) + 1.0f/(8.0f*x));
        else
            return std::log(I0(x));
    }

    static float g(float beta, float theta) {
        return std::exp(-theta*theta/(2.0f*beta*beta))/(std::sqrt(2.0f*M_PI_FLT)*beta);
    }

    static float D(float beta, float phi) {
        float result = 0.0f;
        float delta;
        float shift = 0.0f;
        do {
            delta = g(beta, phi + shift) + g(beta, phi - shift - 2 * M_PI_FLT);
            result += delta;
            shift += 2 * M_PI_FLT;
        } while (delta > 1e-4f);
        return result;
    }

    static float Phi(float gammaI, float gammaT, int p) {
        return 2.0f*p*gammaT - 2.0f*gammaI + p * M_PI_FLT;
    }

    float NrIntegrand(float beta, float halfWiDotWo, float phi, float h) {
        float gammaI = std::asin(math::clamp(h, -1.0f, 1.0f));
        float deltaPhi = phi + 2.0f*gammaI;
        deltaPhi = std::fmod(deltaPhi, 2.0 * M_PI_FLT);
        if (deltaPhi < 0.0f)
            deltaPhi += 2.0 * M_PI_FLT;

        return D(beta, deltaPhi)*fresnelDielectricExt(1.0f/m_eta, halfWiDotWo);
    }

    static Vector3f exp(Vector3f v) {
        return Vector3f(std::exp(v.x), std::exp(v.y), std::exp(v.z));
    }

    Vector3f NpIntegrand(float beta, float cosThetaD, float phi, int p, float h) {
        float iorPrime = std::sqrt(m_eta*m_eta - (1.0f - cosThetaD*cosThetaD))/cosThetaD;
        float cosThetaT = std::sqrt(1.0f - (1.0f - cosThetaD*cosThetaD)*(1.0f/(m_eta * m_eta)));
        Vector3f sigmaAPrime = _sigmaA/cosThetaT;

        float gammaI = std::asin(math::clamp(h, -1.0f, 1.0f));
        float gammaT = std::asin(math::clamp(h/iorPrime, -1.0f, 1.0f));
        // The correct internal path length (the one in d'Eon et al.'s paper
        // as well as Marschner et al.'s paper is wrong).
        // The correct factor is also mentioned in "Light Scattering from Filaments", eq. (20)
        float l = 2.0f*std::cos(gammaT);

        float f = fresnelDielectricExt(1.0f/m_eta, cosThetaD*trigInverse(h));
        Vector3f T = exp(-sigmaAPrime*l);
        Vector3f Aph = (1.0f - f)*(1.0f - f) * T;
        for (int i = 1; i < p; ++i)
            Aph *= f*T;

        float deltaPhi = phi - Phi(gammaI, gammaT, p);
        deltaPhi = std::fmod(deltaPhi, 2.0 * M_PI_FLT);
        if (deltaPhi < 0.0f)
            deltaPhi += 2 * M_PI_FLT;

        return Aph*D(beta, deltaPhi);
    }

    // static float M(float beta, float theta, float alpha) {
    //     return g(beta, theta - alpha);
    // }
    
    static float M(float v, float sinThetaI, float sinThetaO, float cosThetaI, float cosThetaO) {
        float a = cosThetaI*cosThetaO/v;
        float b = sinThetaI*sinThetaO/v;

        if (v < 0.1f)
            // More numerically stable evaluation for small roughnesses
            // See https://publons.com/discussion/12/
            return std::exp(-b + logI0(a) - 1.0f/v + 0.6931f + std::log(1.0f/(2.0f*v)));
        else
            return std::exp(-b)*I0(a)/(2.0f*v*std::sinh(1.0f/v));
    }


    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);
        // bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
        //         && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        // bool sampleTransmission = (bRec.typeMask & ENull)
        //         && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;

        // Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;
        
        // MediumSamplingRecord dummy;
        // PhaseFunctionSamplingRecord pRec(dummy,bRec.wi,bRec.wo);
        // Float phaseVal = m_phase->eval(pRec);

        // // Account for internal reflections: R' = R + TRT + TR^3T + ..
        // if (R < 1)
        //     R += T*T * R / (1-R*R);

        // if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
        //     if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         return Spectrum(0.0f);

        //     return m_specularReflectance->eval(bRec.its) * R * phaseVal;
        // } else {
        //     if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         return Spectrum(0.0f);

        //     return m_specularTransmittance->eval(bRec.its) * (1 - R) * phaseVal;
        // }

        // if (!event.requestedLobe.test(BsdfLobes::GlossyLobe))
        //     return Vec3f(0.0f);

        float sinThetaI = bRec.wi.y, sinThetaO = bRec.wo.y;
        float cosThetaO = trigInverse(sinThetaO);
        float thetaI = std::asin(math::clamp(sinThetaI, -1.0f, 1.0f));
        float thetaO = std::asin(math::clamp(sinThetaO, -1.0f, 1.0f));

        // float thetaI = std::atan2(bRec.wi.z, bRec.wi.x);
        // float thetaO = std::atan2(bRec.wo.z, bRec.wi.x);

        // float phiI = std::atan2(math::safe_sqrt(bRec.wi.x * bRec.wi.x + bRec.wi.z * bRec.wi.z), bRec.wi.y);
        // float phiO = std::atan2(math::safe_sqrt(bRec.wo.x * bRec.wo.x + bRec.wo.z * bRec.wo.z), bRec.wo.y);

        float thetaD = (thetaO - thetaI)*0.5f, thetaH = (thetaO + thetaI)*0.5f;
        float cosThetaD = std::cos(thetaD);

        float phi = std::atan2(bRec.wo.x, bRec.wo.z);
        if (phi < 0.0f)
            phi += M_PI_FLT * 2.0f;

        // Lobe shift due to hair scale tilt, following the values in
        // "Importance Sampling for Physically-Based Hair Fiber Models"
        // rather than the earlier paper by Marschner et al. I believe
        // these are slightly more accurate.
        float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
        float thetaITT  = thetaI +      _scaleAngleRad;
        float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

        // Evaluate longitudinal scattering functions
        // float MR   = M(_vR,   std::sin(thetaH),   sinThetaO, std::cos(thetaH),  cosThetaO);
        // float MTT  = M(_vTT,  std::sin(thetaH),  sinThetaO, std::cos(thetaH),  cosThetaO);
        // float MTRT = M(_vTRT, std::sin(thetaH), sinThetaO, std::cos(thetaH), cosThetaO);

        // float MR   = M(_betaR, thetaH, _scaleAngleRad);
        // float MTT  = M(_betaTT, thetaH, _scaleAngleRad * -0.5f);
        // float MTRT = M(_betaTRT, thetaH, _scaleAngleRad * -1.5f);

        float MR   = M(_vR,   std::sin(thetaIR),   sinThetaO, std::cos(thetaIR),   cosThetaO);
        float MTT  = M(_vTT,  std::sin(thetaITT),  sinThetaO, std::cos(thetaITT),  cosThetaO);
        float MTRT = M(_vTRT, std::sin(thetaITRT), sinThetaO, std::cos(thetaITRT), cosThetaO);

        Vector3f temp = MR*  _nR->eval(phi, cosThetaD)
                +  MTT* _nTT->eval(phi, cosThetaD)
                + MTRT*_nTRT->eval(phi, cosThetaD);

        float value[3] = {temp.x, temp.y, temp.z};

        Spectrum result = Spectrum(value);

        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );  
        if (hasDiffuse) {
            Spectrum diff = m_diffuseReflectance->eval(bRec.its);
            float T12 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());
            float T21 = m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wo), distr.getAlpha());
            float Fdr = 1-m_internalRoughTransmittance->evalDiffuse(distr.getAlpha());

            if (m_nonlinear)
                diff /= Spectrum(1.0f) - diff * Fdr;
            else
                diff /= 1-Fdr;

            result += diff * (INV_PI * Frame::cosTheta(bRec.wo) * T12 * T21 * m_invEta2);
        }

        return result;
    }
    
    static inline float trigInverse(float x) {
        return std::min(std::sqrt(std::max(1.0f - x*x, 0.0f)), 1.0f);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);
        // bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
        //         && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;
        // bool sampleTransmission = (bRec.typeMask & ENull)
        //         && (bRec.component == -1 || bRec.component == 1) && measure == EDiscrete;

        // Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

        // MediumSamplingRecord dummy;
        // PhaseFunctionSamplingRecord pRec(dummy, bRec.wi, bRec.wo);
        // Float pdf = m_phase->pdf(pRec);

        // // Account for internal reflections: R' = R + TRT + TR^3T + ..
        // if (R < 1)
        //     R += T*T * R / (1-R*R);

        // if (Frame::cosTheta(bRec.wi) * Frame::cosTheta(bRec.wo) >= 0) {
        //     if (!sampleReflection || std::abs(dot(reflect(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         return 0.0f;

        //     return pdf * (sampleTransmission ? R : 1.0f);
        // } else {
        //     if (!sampleTransmission || std::abs(dot(transmit(bRec.wi), bRec.wo)-1) > DeltaEpsilon)
        //         return 0.0f;

        //     return pdf * (sampleReflection ? 1-R : 1.0f);
        // }
        float sinThetaI = bRec.wi.y;
        float sinThetaO = bRec.wo.y;
        float cosThetaI = trigInverse(sinThetaI);
        float cosThetaO = trigInverse(sinThetaO);
        float thetaI = std::asin(math::clamp(sinThetaI, -1.0f, 1.0f));
        float thetaO = std::asin(math::clamp(sinThetaO, -1.0f, 1.0f));
        float thetaD = (thetaO - thetaI)*0.5f;
        float cosThetaD = std::cos(thetaD);

        float phi = std::atan2(bRec.wo.x, bRec.wo.z);
        if (phi < 0.0f)
            phi += 2.0 * M_PI_FLT;

        float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
        float thetaITT  = thetaI +      _scaleAngleRad;
        float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

        float weightR   = _nR  ->weight(cosThetaI);
        float weightTT  = _nTT ->weight(cosThetaI);
        float weightTRT = _nTRT->weight(cosThetaI);
        float weightSum = weightR + weightTT + weightTRT;

        float pdfR   = weightR  *M(_vR,   std::sin(thetaIR),   sinThetaO, std::cos(thetaIR),   cosThetaO);
        float pdfTT  = weightTT *M(_vTT,  std::sin(thetaITT),  sinThetaO, std::cos(thetaITT),  cosThetaO);
        float pdfTRT = weightTRT*M(_vTRT, std::sin(thetaITRT), sinThetaO, std::cos(thetaITRT), cosThetaO);

        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );

        /* Calculate the reflection half-vector */
        const Vector H = normalize(bRec.wo+bRec.wi);

        float probDiffuse, probSpecular;
        if (hasDiffuse) {
            /* Find the probability of sampling the specular component */
            probSpecular = 1-m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());

            /* Reallocate samples */
            probSpecular = (probSpecular*m_specularSamplingWeight) /
                (probSpecular*m_specularSamplingWeight +
                (1-probSpecular) * (1-m_specularSamplingWeight));

            probDiffuse = 1 - probSpecular;
        } else {
            probDiffuse = probSpecular = 1.0f;
        }
        
        float result = (1.0f/weightSum)*
        (pdfR  *  _nR->pdf(phi, cosThetaD)
        + pdfTT * _nTT->pdf(phi, cosThetaD)
        + pdfTRT*_nTRT->pdf(phi, cosThetaD)) * probSpecular;        

        if (hasDiffuse)
            result += probDiffuse * warp::squareToCosineHemispherePdf(bRec.wo);

        return result;
    }

    float sampleM(float v, float sinThetaI, float cosThetaI, float xi1, float xi2) const
    {
        // Version from the paper (very unstable)
        //float cosTheta = v*std::log(std::exp(1.0f/v) - 2.0f*xi1*std::sinh(1.0f/v));
        // More stable version from "Numerically stable sampling of the von Mises Fisher distribution on S2 (and other tricks)"
        float cosTheta = 1.0f + v*std::log(xi1 + (1.0f - xi1)*std::exp(-2.0f/v));
        float sinTheta = trigInverse(cosTheta);
        float cosPhi = std::cos(2 * M_PI_FLT *xi2);

        return -cosTheta*sinThetaI + sinTheta*cosPhi*cosThetaI;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, Float &pdf, const Point2 &sample) const {
        bool hasDiffuse = (bRec.typeMask & EDiffuseReflection) &&
            (bRec.component == -1 || bRec.component == 1);
        // bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
        //         && (bRec.component == -1 || bRec.component == 0);
        // bool sampleTransmission = (bRec.typeMask & ENull)
        //         && (bRec.component == -1 || bRec.component == 1);

        // Float R = fresnelDielectricExt(std::abs(Frame::cosTheta(bRec.wi)), m_eta), T = 1-R;

        // PhaseFunctionSamplingRecord pRec(MediumSamplingRecord(), bRec.wi, bRec.wo);
        // m_phase->sample(pRec, pdf, bRec.sampler);

        // // Account for internal reflections: R' = R + TRT + TR^3T + ..
        // if (R < 1)
        //     R += T*T * R / (1-R*R);

        // if (sampleTransmission && sampleReflection) {
        //     if (sample.x <= R) {
        //         bRec.sampledComponent = 0;
        //         bRec.sampledType = EDeltaReflection;
        //         bRec.wo = reflect(bRec.wi);
        //         bRec.eta = 1.0f;
        //         pdf *= R;

        //         return m_specularReflectance->eval(bRec.its);
        //     } else {
        //         bRec.sampledComponent = 1;
        //         bRec.sampledType = ENull;
        //         bRec.wo = transmit(bRec.wi);
        //         bRec.eta = 1.0f;
        //         pdf *= 1-R;

        //         return m_specularTransmittance->eval(bRec.its);
        //     }
        // } else if (sampleReflection) {
        //     bRec.sampledComponent = 0;
        //     bRec.sampledType = EDeltaReflection;
        //     bRec.wo = reflect(bRec.wi);
        //     bRec.eta = 1.0f;
        //     pdf = 1.0f;

        //     return m_specularReflectance->eval(bRec.its) * R;
        // } else if (sampleTransmission) {
        //     bRec.sampledComponent = 1;
        //     bRec.sampledType = ENull;
        //     bRec.wo = transmit(bRec.wi);
        //     bRec.eta = 1.0f;
        //     pdf = 1.0f;

        //     return m_specularTransmittance->eval(bRec.its) * (1-R);
        // }
        // return Spectrum(0.0f);

        Point2 xiN = sample; // TODO should use random samples
        Point2 xiM = sample;

        float sinThetaI = bRec.wi.y;
        float cosThetaI = trigInverse(sinThetaI);
        float thetaI = std::asin(math::clamp(sinThetaI, -1.0f, 1.0f));

        float thetaIR   = thetaI - 2.0f*_scaleAngleRad;
        float thetaITT  = thetaI +      _scaleAngleRad;
        float thetaITRT = thetaI + 4.0f*_scaleAngleRad;

        // The following lines are just lobe selection
        float weightR   = _nR  ->weight(cosThetaI);
        float weightTT  = _nTT ->weight(cosThetaI);
        float weightTRT = _nTRT->weight(cosThetaI);

        const Azimuthal *lobe;
        float v;
        float theta;

        float target = xiN.x*(weightR + weightTT + weightTRT);
        if (target < weightR) {
            v = _vR;
            theta = thetaIR;
            lobe = _nR.get();
        } else if (target < weightR + weightTT) {
            v = _vTT;
            theta = thetaITT;
            lobe = _nTT.get();
        } else {
            v = _vTRT;
            theta = thetaITRT;
            lobe = _nTRT.get();
        }

        // Actual sampling of the direction starts here
        float sinThetaO = sampleM(v, std::sin(theta), std::cos(theta), xiM.x, xiM.y);
        float cosThetaO = trigInverse(sinThetaO);

        float thetaO = std::asin(math::clamp(sinThetaO, -1.0f, 1.0f));
        float thetaD = (thetaO - thetaI)*0.5f;
        float cosThetaD = std::cos(thetaD);

        float phi, phiPdf;
        lobe->sample(cosThetaD, xiN.y, phi, phiPdf);

        float sinPhi = std::sin(phi);
        float cosPhi = std::cos(phi);

        MicrofacetDistribution distr(
            m_type,
            m_alpha->eval(bRec.its).average(),
            m_sampleVisible
        );

        bool choseSpecular = true;
        float probSpecular;
        if (hasDiffuse) {
            /* Find the probability of sampling the specular component */
            probSpecular = 1 - m_externalRoughTransmittance->eval(Frame::cosTheta(bRec.wi), distr.getAlpha());

            /* Reallocate samples */
            probSpecular = (probSpecular*m_specularSamplingWeight) /
                (probSpecular*m_specularSamplingWeight +
                (1-probSpecular) * (1-m_specularSamplingWeight));

            if (sample.y < probSpecular) {
            } else {
                choseSpecular = false;
            }
        }

        if (choseSpecular) {
            /* Perfect specular reflection based on the microfacet normal */
            bRec.wo = Vector3f(sinPhi*cosThetaO, sinThetaO, cosPhi*cosThetaO);
            pdf = Marschner::pdf(bRec, ESolidAngle); // TODO vary esolidangle
            // bRec.weight = eval(event)/event.pdf;// TODO watch out for the weight
            bRec.sampledType = EDeltaReflection;
        } else {
            bRec.sampledComponent = 1;
            bRec.sampledType = EDiffuseReflection;
            bRec.wo = warp::squareToCosineHemisphere(sample);
        }
        bRec.eta = 1.0f;

        /* Guard against numerical imprecisions */
        pdf = Marschner::pdf(bRec, ESolidAngle);

        if (pdf == 0)
            return Spectrum(0.0f);
        else
            return eval(bRec, ESolidAngle) / pdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        float pdf;
        return Marschner::sample(bRec, pdf, sample);
    }

    void precomputeAzimuthalDistributions() {
        const int Resolution = Azimuthal::AzimuthalResolution;
        std::unique_ptr<Vector3f[]> valuesR  (new Vector3f[Resolution*Resolution]);
        std::unique_ptr<Vector3f[]> valuesTT (new Vector3f[Resolution*Resolution]);
        std::unique_ptr<Vector3f[]> valuesTRT(new Vector3f[Resolution*Resolution]);

        // Ideally we could simply make this a constexpr, but MSVC does not support that yet (boo!)
        #define NumPoints 140

        GaussLegendre<NumPoints> integrator;
        const auto points = integrator.points();
        const auto weights = integrator.weights();

        // Cache the gammaI across all integration points
        std::array<float, NumPoints> gammaIs;
        for (int i = 0; i < NumPoints; ++i)
            gammaIs[i] = std::asin(points[i]);

        // Precompute the Gaussian detector and sample it into three 1D tables.
        // This is the only part of the precomputation that is actually approximate.
        // 2048 samples are enough to support the lowest roughness that the BCSDF
        // can reliably simulate
        const int NumGaussianSamples = 2048;
        std::unique_ptr<float[]> Ds[3];
        for (int p = 0; p < 3; ++p) {
            Ds[p].reset(new float[NumGaussianSamples]);
            for (int i = 0; i < NumGaussianSamples; ++i)
                Ds[p][i] = D(_betaR, i/(NumGaussianSamples - 1.0f) * 2 * M_PI_FLT);
        }

        // Simple wrapped linear interpolation of the precomputed table
        auto approxD = [&](int p, float phi) {
            float u = std::abs(phi*(1.0 / (2 * M_PI_FLT) *(NumGaussianSamples - 1)));
            int x0 = int(u);
            int x1 = x0 + 1;
            u -= x0;
            return Ds[p][x0 % NumGaussianSamples]*(1.0f - u) + Ds[p][x1 % NumGaussianSamples]*u;
        };

        // Here follows the actual precomputation of the azimuthal scattering functions
        // The scattering functions are parametrized with the azimuthal angle phi,
        // and the cosine of the half angle, cos(thetaD).
        // This parametrization makes the azimuthal function relatively smooth and allows using
        // really low resolutions for the table (64x64 in this case) without any visual
        // deviation from ground truth, even at the lowest supported roughness setting
        for (int y = 0; y < Resolution; ++y) {
            float cosHalfAngle = y/(Resolution - 1.0f);

            // Precompute reflection Fresnel factor and reduced absorption coefficient
            float iorPrime = std::sqrt(m_eta*m_eta - (1.0f - cosHalfAngle*cosHalfAngle))/cosHalfAngle;
            float cosThetaT = std::sqrt(1.0f - (1.0f - cosHalfAngle*cosHalfAngle)*(1.0f/m_eta)*(1.0f/m_eta));
            Vector3f sigmaAPrime = _sigmaA/cosThetaT;

            // Precompute gammaT, f_t and internal absorption across all integration points
            std::array<float, NumPoints> fresnelTerms, gammaTs;
            std::array<Vector3f, NumPoints> absorptions;
            for (int i = 0; i < NumPoints; ++i) {
                gammaTs[i] = std::asin(math::clamp(points[i]/iorPrime, -1.0f, 1.0f));
                fresnelTerms[i] = fresnelDielectricExt(1.0f/m_eta, cosHalfAngle*std::cos(gammaIs[i]));
                absorptions[i] = exp(-sigmaAPrime*2.0f*std::cos(gammaTs[i]));
            }

            for (int phiI = 0; phiI < Resolution; ++phiI) {
                float phi = M_PI_FLT * 2 *phiI/(Resolution - 1.0f);

                float integralR = 0.0f;
                Vector3f integralTT(0.0f);
                Vector3f integralTRT(0.0f);

                // Here follows the integration across the fiber width, h.
                // Since we were able to precompute most of the factors that
                // are constant w.r.t phi for a given h,
                // we don't have to do much work here.
                for (int i = 0; i < integrator.numSamples(); ++i) {
                    float fR = fresnelTerms[i];
                    Vector3f T = absorptions[i];

                    float AR = fR;
                    Vector3f ATT = (1.0f - fR)*(1.0f - fR)*T;
                    Vector3f ATRT = ATT*fR*T;

                    integralR   += weights[i]*approxD(0, phi - Phi(gammaIs[i], gammaTs[i], 0))*AR;
                    integralTT  += weights[i]*approxD(1, phi - Phi(gammaIs[i], gammaTs[i], 1))*ATT;
                    integralTRT += weights[i]*approxD(2, phi - Phi(gammaIs[i], gammaTs[i], 2))*ATRT;
                }

                valuesR  [phiI + y*Resolution] = Vector3f(0.5f*integralR);
                valuesTT [phiI + y*Resolution] = 0.5f*integralTT;
                valuesTRT[phiI + y*Resolution] = 0.5f*integralTRT;
            }
        }

        // Hand the values off to the helper class to construct sampling CDFs and so forth
        _nR  .reset(new Azimuthal(std::move(valuesR)));
        _nTT .reset(new Azimuthal(std::move(valuesTT)));
        _nTRT.reset(new Azimuthal(std::move(valuesTRT)));
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
        oss << "ThinDielectric[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  eta = " << m_eta << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularTransmittance = " << indent(m_specularTransmittance->toString()) << endl
            << "  phase = " << indent(m_phase->toString()) << "," << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularTransmittance;
    ref<Texture> m_specularReflectance;
    ref<PhaseFunction> m_phase;
    float m_eta;
    float _betaR;
    float _betaTT;
    float _betaTRT;
    float _roughness;
    float _scaleAngleRad;
    Vector3f _sigmaA;
    std::unique_ptr<Azimuthal> _nR, _nTT, _nTRT;
    float _vR, _vTT, _vTRT;

    MicrofacetDistribution::EType m_type;
    ref<RoughTransmittance> m_externalRoughTransmittance;
    ref<RoughTransmittance> m_internalRoughTransmittance;
    ref<Texture> m_diffuseReflectance;
    ref<Texture> m_alpha;
    float m_invEta2;
    float m_specularSamplingWeight;
    bool m_nonlinear;
    bool m_sampleVisible;
};

/* Fake glass shader -- it is really hopeless to visualize
   this material in the VPL renderer, so let's try to do at least
   something that suggests the presence of a transparent boundary */
class MarschnerShader : public Shader {
public:
    MarschnerShader(Renderer *renderer) :
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

Shader *Marschner::createShader(Renderer *renderer) const {
    return new MarschnerShader(renderer);
}

MTS_IMPLEMENT_CLASS(MarschnerShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(Marschner, false, BSDF)
MTS_EXPORT_PLUGIN(Marschner, "Thin dielectric BSDF");
MTS_NAMESPACE_END
