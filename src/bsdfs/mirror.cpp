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
#include "microfacet.h"
#include "rtrans.h"
#include "ior.h"

MTS_NAMESPACE_BEGIN

/*!\plugin{roughplastic}{Rough plastic material}
 * \order{9}
 * \icon{bsdf_roughplastic}
 * \parameters{
 *     \parameter{distribution}{\String}{
 *          Specifies the type of microfacet normal distribution
 *          used to model the surface roughness.
 *          \vspace{-1mm}
 *       \begin{enumerate}[(i)]
 *           \item \code{beckmann}: Physically-based distribution derived from
 *               Gaussian random surfaces. This is the default.\vspace{-1.5mm}
 *           \item \code{ggx}: The GGX \cite{Walter07Microfacet} distribution (also known as
 *               Trowbridge-Reitz \cite{Trowbridge19975Average} distribution)
 *               was designed to better approximate the long tails observed in measurements
 *               of ground surfaces, which are not modeled by the Beckmann distribution.
 *           \vspace{-1.5mm}
 *           \item \code{phong}: Classical Phong distribution.
 *              In most cases, the \code{ggx} and \code{beckmann} distributions
 *              should be preferred, since they provide better importance sampling
 *              and accurate shadowing/masking computations.
 *              \vspace{-4mm}
 *       \end{enumerate}
 *     }
 *     \parameter{alpha}{\Float\Or\Texture}{
 *         Specifies the roughness of the unresolved surface micro-geometry.
 *         When the Beckmann distribution is used, this parameter is equal to the
 *         \emph{root mean square} (RMS) slope of the microfacets.
 *         \default{0.1}.
 *     }
 *
 *     \parameter{intIOR}{\Float\Or\String}{Interior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{polypropylene} / 1.49}}
 *     \parameter{extIOR}{\Float\Or\String}{Exterior index of refraction specified
 *      numerically or using a known material name. \default{\texttt{air} / 1.000277}}
 *     \parameter{sampleVisible}{\Boolean}{
 *         Enables an improved importance sampling technique. Refer to
 *         pages \pageref{plg:roughconductor} and \pageref{sec:visiblenormal-sampling}
 *         for details. \default{\code{true}}
 *     }
 *     \parameter{specular\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor that can be used to modulate the specular reflection component. Note
 *         that for physical realism, this parameter should never be touched. \default{1.0}}
 *     \parameter{diffuse\showbreak Reflectance}{\Spectrum\Or\Texture}{Optional
 *         factor used to modulate the diffuse reflection component\default{0.5}}
 *     \parameter{nonlinear}{\Boolean}{
 *         Account for nonlinear color shifts due to internal scattering? See the
 *         \pluginref{plastic} plugin for details.\default{Don't account for them and
 *         preserve the texture colors, i.e. \code{false}}
 *     }
 * }
 *
 * \vspace{3mm}
 * This plugin implements a realistic microfacet scattering model for rendering
 * rough dielectric materials with internal scattering, such as plastic. It can
 * be interpreted as a fancy version of the Cook-Torrance model and should be
 * preferred over heuristic models like \pluginref{phong} and \pluginref{ward}
 * when possible.
 *
 * Microfacet theory describes rough surfaces as an arrangement of
 * unresolved and ideally specular facets, whose normal directions are given by
 * a specially chosen \emph{microfacet distribution}. By accounting for shadowing
 * and masking effects between these facets, it is possible to reproduce the important
 * off-specular reflections peaks observed in real-world measurements of such
 * materials.
 *
 * \renderings{
 *     \rendering{Beckmann, $\alpha=0.1$}{bsdf_roughplastic_beckmann}
 *     \rendering{GGX, $\alpha=0.3$}{bsdf_roughplastic_ggx}
 * }
 *
 * This plugin is essentially the ``roughened'' equivalent of the (smooth) plugin
 * \pluginref{plastic}. For very low values of $\alpha$, the two will
 * be identical, though scenes using this plugin will take longer to render
 * due to the additional computational burden of tracking surface roughness.
 *
 * For convenience, this model allows to specify IOR values either numerically,
 * or based on a list of known materials (see \tblref{dielectric-iors} on
 * \tblpage{dielectric-iors} for an overview).
 * When no parameters are given, the plugin activates the defaults,
 * which describe a white polypropylene plastic material with a light amount
 * of roughness modeled using the Beckmann distribution.
 *
 * Like the \pluginref{plastic} material, this model internally simulates the
 * interaction of light with a diffuse base surface coated by a thin dielectric
 * layer (where the coating layer is now \emph{rough}). This is a convenient
 * abstraction rather than a restriction. In other words, there are many
 * materials that can be rendered with this model, even if they might not
 * fit this description perfectly well.
 *
 * The simplicity of this setup makes it possible to account for interesting
 * nonlinear effects due to internal scattering, which is controlled by
 * the \texttt{nonlinear} parameter. For more details, please refer to the description
 * of this parameter given in the \pluginref{plastic} plugin section
 * on \pluginpage{plastic}.
 *
 * To get an intuition about the effect of the surface roughness parameter
 * $\alpha$, consider the following approximate classification: a value of
 * $\alpha=0.001-0.01$ corresponds to a material with slight imperfections
 * on an otherwise smooth surface finish, $\alpha=0.1$ is relatively rough,
 * and $\alpha=0.3-0.7$ is \emph{extremely} rough (e.g. an etched or ground
 * finish). Values significantly above that are probably not too realistic.
 *
 * \renderings{
 *     \medrendering{Diffuse textured rendering}{bsdf_plastic_diffuse}
 *     \medrendering{Textured rough plastic model and \code{nonlinear=false}}{bsdf_roughplastic_preserve}
 *     \medrendering{Textured rough plastic model and \code{nonlinear=true}}{bsdf_roughplastic_nopreserve}
 *     \caption{
 *        When asked to do so, this model can account for subtle nonlinear color shifts due
 *        to internal scattering processes. The above images show a textured
 *        object first rendered using \pluginref{diffuse}, then
 *        \pluginref{roughplastic} with the default parameters, and finally using
 *        \pluginref{roughplastic} and support for nonlinear color shifts.
 *     }
 * }
 * \renderings{
 *     \rendering{Wood material with smooth horizontal stripes}{bsdf_roughplastic_roughtex1}
 *     \rendering{A material with imperfections at a much smaller scale than what
 *       is modeled e.g. using a bump map.}{bsdf_roughplastic_roughtex2}\vspace{-3mm}
 *     \caption{
 *         The ability to texture the roughness parameter makes it possible
 *         to render materials with a structured finish, as well as
 *         ``smudgy'' objects.
 *     }
 * }
 * \vspace{2mm}
 * \begin{xml}[caption={A material definition for black plastic material with
 *    a spatially varying roughness.},
 *    label=lst:roughplastic-varyingalpha]
 * <bsdf type="roughplastic">
 *     <string name="distribution" value="beckmann"/>
 *     <float name="intIOR" value="1.61"/>
 *     <spectrum name="diffuseReflectance" value="0"/>
 *     <!-- Fetch roughness values from a texture and slightly reduce them -->
 *     <texture type="scale" name="alpha">
 *         <texture name="alpha" type="bitmap">
 *             <string name="filename" value="roughness.png"/>
 *         </texture>
 *         <float name="scale" value="0.6"/>
 *     </texture>
 * </bsdf>
 * \end{xml}
 *
 * \subsubsection*{Technical details}
 * The implementation of this model is partly based on the paper ``Microfacet
 * Models for Refraction through Rough Surfaces'' by Walter et al.
 * \cite{Walter07Microfacet}. Several different types of microfacet
 * distributions are supported. Note that the choices are slightly more
 * restricted here---in comparison to other rough scattering models in
 * Mitsuba, anisotropic distributions are not allowed.
 *
 * The implementation of this model makes heavy use of a \emph{rough
 * Fresnel transmittance} function, which is a generalization of the
 * usual Fresnel transmittion coefficient to microfacet surfaces. Unfortunately,
 * this function is normally prohibitively expensive, since each
 * evaluation involves a numerical integration over the sphere.
 *
 * To avoid this performance issue, Mitsuba ships with data files
 * (contained in the \code{data/microfacet} directory) containing precomputed
 * values of this function over a large range of parameter values. At runtime,
 * the relevant parts are extracted using tricubic interpolation.
 *
 * When rendering with the Phong microfacet distribution, a conversion is
 * used to turn the specified Beckmann-equivalent $\alpha$ roughness value
 * into the exponent parameter of this distribution. This is done in a way,
 * such that the same value $\alpha$ will produce a similar appearance across
 * different microfacet distributions.
 */
class Mirror : public BSDF {
public:
    Mirror(const Properties &props) : BSDF(props) {
        m_specularReflectance = new ConstantSpectrumTexture(
            props.getSpectrum("specularReflectance", Spectrum(1.0f)));
    }

    Mirror(Stream *stream, InstanceManager *manager)
     : BSDF(stream, manager) {
        m_specularReflectance = static_cast<Texture *>(manager->getInstance(stream));

        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        BSDF::serialize(stream, manager);
        
        manager->serialize(stream, m_specularReflectance.get());
    }

    void configure() {
        m_specularReflectance = ensureEnergyConservation(
            m_specularReflectance, "specularReflectance", 1.0f);
        
        m_components.clear();
        m_components.push_back(EDeltaReflection | EFrontSide | EBackSide
            | (m_specularReflectance->isConstant() ? 0 : ESpatiallyVarying));

        m_usesRayDifferentials =
            m_specularReflectance->usesRayDifferentials();

        BSDF::configure();
    }

    Spectrum getDiffuseReflectance(const Intersection &its) const {
        return Spectrum(0.0f);
    }

    Spectrum getSpecularReflectance(const Intersection &its) const {
        return m_specularReflectance->eval(its);
    }

    /// Reflection in local coordinates
    inline Vector reflect(const Vector &wi) const {
        return Vector(-wi.x, -wi.y, wi.z);
    }

    Spectrum eval(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool sampleReflection   = (bRec.typeMask & EDeltaReflection)
            && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;

        if (measure != ESolidAngle ||
            Frame::cosTheta(bRec.wi) <= 0 ||
            Frame::cosTheta(bRec.wo) <= 0 ||
            (!sampleReflection))
            return Spectrum(0.0f);

        return m_specularReflectance->eval(bRec.its);
    }

    Float pdf(const BSDFSamplingRecord &bRec, EMeasure measure) const {
        bool hasSpecular = (bRec.typeMask & EDeltaReflection)
                && (bRec.component == -1 || bRec.component == 0) && measure == EDiscrete;

        return hasSpecular ? 1 : 0;
    }

    inline Spectrum sample(BSDFSamplingRecord &bRec, Float &_pdf, const Point2 &_sample) const {
        bool hasSpecular = (bRec.typeMask & EDeltaReflection) &&
            (bRec.component == -1 || bRec.component == 0);

        // printf("%d %d \n", hasSpecular, (bRec.typeMask & EDeltaReflection));

        if (Frame::cosTheta(bRec.wi) <= 0 || (!hasSpecular))
            return Spectrum(0.0f);

        if (hasSpecular) {
            bRec.sampledComponent = 0;
            bRec.sampledType = EDeltaReflection;
            bRec.wo = reflect(bRec.wi);
            bRec.eta = 1.0f;
            _pdf = 1.0f;

            return m_specularReflectance->eval(bRec.its);
        }

        return Spectrum(0.0f);
    }

    Spectrum sample(BSDFSamplingRecord &bRec, const Point2 &sample) const {
        Float pdf;
        return Mirror::sample(bRec, pdf, sample);
    }

    void addChild(const std::string &name, ConfigurableObject *child) {
        if (child->getClass()->derivesFrom(MTS_CLASS(Texture))) {
            if (name == "specularReflectance")
                m_specularReflectance = static_cast<Texture *>(child);
            else
                BSDF::addChild(name, child);
        } else {
            BSDF::addChild(name, child);
        }
    }

    Float getRoughness(const Intersection &its, int component) const {
        return 0.0f;
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "Mirror[" << endl
            << "  id = \"" << getID() << "\"," << endl
            << "  specularReflectance = " << indent(m_specularReflectance->toString()) << "," << endl
            << "]";
        return oss.str();
    }

    Shader *createShader(Renderer *renderer) const;

    MTS_DECLARE_CLASS()
private:
    ref<Texture> m_specularReflectance;
};

/**
 * GLSL port of the rough plastic shader. This version is much more
 * approximate -- it only supports the Beckmann distribution,
 * does everything in RGB, uses a cheaper shadowing-masking term, and
 * it also makes use of the Schlick approximation to the Fresnel
 * reflectance of dielectrics. When the roughness is lower than
 * \alpha < 0.2, the shader clamps it to 0.2 so that it will still perform
 * reasonably well in a VPL-based preview. There is no support for
 * non-linear effects due to internal scattering.
 */
class MirrorShader : public Shader {
public:
    MirrorShader(Renderer *renderer) :
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

Shader *Mirror::createShader(Renderer *renderer) const {
    return new MirrorShader(renderer);
}

MTS_IMPLEMENT_CLASS(MirrorShader, false, Shader)
MTS_IMPLEMENT_CLASS_S(Mirror, false, BSDF)
MTS_EXPORT_PLUGIN(Mirror, "Mirror BRDF");
MTS_NAMESPACE_END
