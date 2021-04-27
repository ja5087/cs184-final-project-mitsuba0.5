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
#include <mitsuba/core/warp.h>

MTS_NAMESPACE_BEGIN

/*!\plugin{kajiyakay}{Kajiya Kay BSDF}
 * \order{14}
 * \parameters{
 *     \parameter{exponent}{\Float\Or\Texture}{
 *         Specifies the Phong exponent \default{30}.
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{
 *         Specifies the weight of the specular reflectance component.\default{0.2}}
 *     \parameter{diffuse\showbreak Reflectance}{\Spectrum\Or\Texture}{
 *         Specifies the weight of the diffuse reflectance component\default{0.5}}
 * }
 * \renderings{
 *     \rendering{Exponent$\,=60$}{bsdf_phong_60}
 *     \rendering{Exponent$\,=300$}{bsdf_phong_300}
 * }

 * TO BE WRITTEN!
 * This plugin implements the modified Phong reflectance model as described in
 * \cite{Phong1975Illumination} and \cite{Lafortune1994Using}. This heuristic
 * model is mainly included for historical reasons---its use in new scenes is
 * discouraged, since significantly more realistic models have been developed
 * since 1975.
 *
 * If possible, it is recommended to switch to a BRDF that is based on
 * microfacet theory and includes knowledge about the material's index of
 * refraction. In Mitsuba, two good alternatives to \pluginref{phong} are
 * the plugins \pluginref{roughconductor} and \pluginref{roughplastic}
 * (depending on the material type).
 *
 * When using this plugin, note that the diffuse and specular reflectance
 * components should add up to a value less than or equal to one (for each
 * color channel). Otherwise, they will automatically be scaled appropriately
 * to ensure energy conservation.
 */
class KajiyaKay : public BSDF {
public:
    KajiyaKay(const Properties &props)
        : BSDF(props) {
        m_diffuseReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("diffuseReflectance", Spectrum(0.5f)));
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(0.2f)));
        m_exponent = new ConstantFloatTexture(
            props.getFloat("exponent", 5.0f));
        m_specularSamplingWeight = 0.0f;
    }

    KajiyaKay(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_diffuseReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));
        m_exponent = static_cast<Texture *>(manager->getInstance(stream));

        configure();
    }

    void configure() {
        m_components.clear();
        m_components.push_back(EGlossyReflection | EFrontSide |
            ((!m_specularReflectance->isConstant()
              || !m_exponent->isConstant()) ? ESpatiallyVarying : 0));
        m_components.push_back(EDiffuseReflection | EFrontSide
            | (m_diffuseReflectance->isConstant() ? 0 : ESpatiallyVarying));

        /* Verify the input parameters and fix them if necessary */
        std::pair<Texture *, Texture *> result = ensureEnergyConservation(
            m_specularReflectance, m_diffuseReflectance,
            "specularReflectance", "diffuseReflectance", 1.0f);
        m_specularReflectance = result.first;
        m_diffuseReflectance = result.second;

        /* Compute weights that steer samples towards
           the specular or diffuse components */
        Float dAvg = m_diffuseReflectance->getAverage().getLuminance(),
              sAvg = m_specularReflectance->getAverage().getLuminance();
        m_specularSamplingWeight = sAvg / (dAvg + sAvg);

        m_usesRayDifferentials =
            m_diffuseReflectance->usesRayDifferentials() ||
            m_specularReflectance->usesRayDifferentials() ||
            m_exponent->usesRayDifferentials();

        BSDF::configure();
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return m_diffuseReflectance->eval(its);
    }

    Spectrum getSpecularReflectance(const Intersection &its) const {
        return m_specularReflectance->eval(its);
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
            return Spectrum(0.0f);

        bool hasSpecular = (bRec.typeMask & EGlossyReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 1);

        Spectrum result(0.0f);
        if (hasSpecular) {
            Vector l = reflect(bRec.wi);
            Float sin_nl = std::sqrt(1-l.z*l.z);
            Float sin_ne = std::sqrt(1-bRec.wo.z*bRec.wo.z);
            Float alpha = (l.z * bRec.wo.z - sin_nl*sin_ne);
            Float exponent = m_exponent->eval(bRec.its).average();
            if (alpha > 0.0f) {
                result += 10.0f * m_specularReflectance->eval(bRec.its) * std::pow(alpha, exponent);
            }
        }

        if (hasDiffuse)
            result += 0.2f * m_diffuseReflectance->eval(bRec.its);

        return result * Frame::cosTheta(bRec.wo);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        if (Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 || measure != ESolidAngle)
            return 0.0f;

        bool hasSpecular = (bRec.typeMask & EGlossyReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 1);

        Float diffuseProb = 0.0f, specProb = 0.0f;

        if (hasDiffuse)
            diffuseProb = warp::squareToCosineHemispherePdf(bRec.wo);

        // // Don't sample specular that's too low to avoid numerical error
        // if (hasSpecular && (bRec.wo.z > -1e-4 && bRec.wo.z < 1e-4)) return 0.0f;

        if (hasSpecular) {
            // specProb = INV_TWOPI_FLT / bRec.wi.z;
            specProb = warp::squareToCosineHemispherePdf(bRec.wo); // Cosine Hemi Sampling
        }

        if (hasDiffuse && hasSpecular)
            return m_specularSamplingWeight * specProb +
                   (1-m_specularSamplingWeight) * diffuseProb;
        else if (hasDiffuse)
            return diffuseProb;
        else if (hasSpecular)
            return specProb;
        else
            return 0.0f;
    }

    inline Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
        Point2 sample(_sample);

        bool hasSpecular = (bRec.typeMask & EGlossyReflection)
                && (bRec.component == -1 || bRec.component == 0);
        bool hasDiffuse  = (bRec.typeMask & EDiffuseReflection)
                && (bRec.component == -1 || bRec.component == 1);

        if (!hasSpecular && !hasDiffuse)
            return Spectrum(0.0f);

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
            // // Sample from a circle of radius z centered around (0,0,height)
            // Float height =  std::sqrt(bRec.wi.x * bRec.wi.x + bRec.wi.y * bRec.wi.y);
            // Float radius = bRec.wi.z;
            // Float theta = (2.0f * M_PI) * sample.x;

            // Vector localDir = Vector(
            //     radius * std::cos(theta),
            //     radius * std::sin(theta),
            //     height
            // );
            // localDir /= localDir.length();

            // // Rotate back into correct coordinate system
            // Vector tangent = reflect(bRec.wi);
            // tangent.z = 0; // Project onto floor
            // bRec.wo = Frame(tangent).toWorld(localDir);
            // Just do cosine hemisphere sampling lol
            bRec.wo = warp::squareToCosineHemisphere(sample);
            bRec.sampledComponent = 1;
            bRec.sampledType = EGlossyReflection;
        } else {
            bRec.wo = warp::squareToCosineHemisphere(sample);
            bRec.sampledComponent = 0;
            bRec.sampledType = EDiffuseReflection;
        }
        bRec.eta = 1.0f;

        _pdf = pdf(bRec, ESolidAngle);

        if (_pdf == 0)
            return Spectrum(0.0f);
        else
            return eval(bRec, ESolidAngle) / _pdf;
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        Float pdf;
        return KajiyaKay::sample(bRec, pdf, sample);
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "exponent")
                m_exponent = static_cast<Texture *>(child);
            else if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else if (name == "diffuseReflectance")
                m_diffuseReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);

        manager->serialize(stream, m_diffuseReflectance.get());
        manager->serialize(stream, m_specularReflectance.get());
        manager->serialize(stream, m_exponent.get());
    }

    Float getRoughness(const Intersection &its, int component) const {
        Assert(component == 0 || component == 1);
        /* Find the Beckmann-equivalent roughness */
        if (component == 0)
            return std::sqrt(2 / (2+m_exponent->eval(its).average()));
        else
            return std::numeric_limits<Float>::infinity();
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "KajiyaKay[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  diffuseReflectance = " << indent(m_diffuseReflectance->toString()) << "," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "  specularSamplingWeight = " << m_specularSamplingWeight << "," << endl
            << "  diffuseSamplingWeight = " << (1-m_specularSamplingWeight) << "," << endl
            << "  exponent = " << indent(m_exponent->toString()) << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_diffuseReflectance;
    ref<Texture> m_specularReflectance;
    ref<Texture> m_exponent;
    Float m_specularSamplingWeight;
};

// ================ Hardware shader implementation ================

/**
 * The GLSL implementation clamps the exponent to 30 so that a
 * VPL renderer will able to handle the material reasonably well.
 * This is based off the old Phong code since we're doing illegal things
 */
class KajiyaKayShader : public Shader {
public:
    KajiyaKayShader(Renderer *renderer, const Texture *exponent,
            const Texture *diffuseColor, const Texture *specularColor)
          : Shader(renderer, EBSDFShader),
            m_exponent(exponent),
            m_diffuseReflectance(diffuseColor),
            m_specularReflectance(specularColor) {
        m_exponentShader = renderer->registerShaderForResource(m_exponent.get());
        m_diffuseReflectanceShader = renderer->registerShaderForResource(m_diffuseReflectance.get());
        m_specularReflectanceShader = renderer->registerShaderForResource(m_specularReflectance.get());
    }

    bool isComplete() const {
        return m_exponentShader.get() != NULL &&
               m_diffuseReflectanceShader.get() != NULL &&
               m_specularReflectanceShader.get() != NULL;
    }

    void putDependencies(std::vector<Shader *> &deps) {
        deps.push_back(m_exponentShader.get());
        deps.push_back(m_diffuseReflectanceShader.get());
        deps.push_back(m_specularReflectanceShader.get());
    }

    void cleanup(Renderer *renderer) {
        renderer->unregisterShaderForResource(m_exponent.get());
        renderer->unregisterShaderForResource(m_diffuseReflectance.get());
        renderer->unregisterShaderForResource(m_specularReflectance.get());
    }

    void generateCode(std::ostringstream &oss,
            const std::string &evalName,
            const std::vector<std::string> &depNames) const {
        oss << "vec3 " << evalName << "(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (cosTheta(wi) <= 0.0 || cosTheta(wo) <= 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    vec3 R = vec3(-wi.x, -wi.y, wi.z);" << endl
            << "    float specRef = 0.0, alpha = dot(R, wo);" << endl
            << "    float exponent = min(30.0, " << depNames[0] << "(uv)[0]);" << endl
            << "    if (alpha > 0.0)" << endl
            << "        specRef = pow(alpha, exponent) * " << endl
            << "      (exponent + 2) * 0.15915;" << endl
            << "    return (" << depNames[1] << "(uv) * inv_pi" << endl
            << "           + " << depNames[2] << "(uv) * specRef) * cosTheta(wo);" << endl
            << "}" << endl
            << "vec3 " << evalName << "_diffuse(vec2 uv, vec3 wi, vec3 wo) {" << endl
            << "    if (wi.z <= 0.0 || wo.z <= 0.0)" << endl
            << "        return vec3(0.0);" << endl
            << "    return " << depNames[1] << "(uv) * (inv_pi * cosTheta(wo));" << endl
            << "}" << endl;
    }


    MTS_DECLARE_CLASS()
private:
    ref<const Texture> m_exponent;
    ref<const Texture> m_diffuseReflectance;
    ref<const Texture> m_specularReflectance;
    ref<Shader> m_exponentShader;
    ref<Shader> m_diffuseReflectanceShader;
    ref<Shader> m_specularReflectanceShader;
};

Shader *KajiyaKay::createShader(Renderer *renderer) const {
    return new KajiyaKayShader(renderer, m_exponent.get(),
        m_diffuseReflectance.get(), m_specularReflectance.get());
}

MTS_IMPLEMENT_CLASS(KajiyaKayShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(KajiyaKay, false, BSDF)
MTS_EXPORT_PLUGIN(KajiyaKay, "Kajiya Kay Hair BSDF");
MTS_NAMESPACE_END
