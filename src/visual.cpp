#include "nori/pathgraph.h"
#include <iostream>

#include <nori/warp.h>
#include <nori/bsdf.h>
#include <nori/bitmap.h>
#include <nori/vector.h>
#include <nanogui/screen.h>
#include <nanogui/label.h>
#include <nanogui/button.h>
#include <nanogui/window.h>
#include <nanogui/layout.h>
#include <nanogui/icons.h>
#include <nanogui/combobox.h>
#include <nanogui/slider.h>
#include <nanogui/textbox.h>
#include <nanogui/checkbox.h>
#include <nanogui/messagedialog.h>
#include <nanogui/renderpass.h>
#include <nanogui/shader.h>
#include <nanogui/texture.h>
#include <nanogui/screen.h>
#include <nanogui/opengl.h>
#include <nanogui/window.h>
#include <nanogui/canvas.h>
#include <nanogui/imageview.h>

#include <nanovg_gl.h>

#include <pcg32.h>
#include <hypothesis.h>
#include <tinyformat.h>

#include <Eigen/Geometry>

#if defined(_MSC_VER)
#  pragma warning (disable: 4305 4244)
#endif

using namespace nanogui;
using namespace std;
#define Float_Infinity std::numeric_limits<float>::infinity;

struct Arcball {
    using Quaternionf = Eigen::Quaternion<float, Eigen::DontAlign>;

    Arcball(float speedFactor = 2.0f)
        : m_active(false), m_lastPos(nori::Vector2i::Zero()), m_size(nori::Vector2i::Zero()),
          m_quat(Quaternionf::Identity()),
          m_incr(Quaternionf::Identity()),
          m_speedFactor(speedFactor) { }

    void setSize(nori::Vector2i size) { m_size = size; }

    const nori::Vector2i &size() const { return m_size; }

    void button(nori::Vector2i pos, bool pressed) {
        m_active = pressed;
        m_lastPos = pos;
        if (!m_active)
            m_quat = (m_incr * m_quat).normalized();
        m_incr = Quaternionf::Identity();
    }

    bool motion(nori::Vector2i pos) {
        if (!m_active)
            return false;

        /* Based on the rotation controller from AntTweakBar */
        float invMinDim = 1.0f / m_size.minCoeff();
        float w = (float) m_size.x(), h = (float) m_size.y();

        float ox = (m_speedFactor * (2*m_lastPos.x() - w) + w) - w - 1.0f;
        float tx = (m_speedFactor * (2*pos.x()      - w) + w) - w - 1.0f;
        float oy = (m_speedFactor * (h - 2*m_lastPos.y()) + h) - h - 1.0f;
        float ty = (m_speedFactor * (h - 2*pos.y())      + h) - h - 1.0f;

        ox *= invMinDim; oy *= invMinDim;
        tx *= invMinDim; ty *= invMinDim;

        nori::Vector3f v0(ox, oy, 1.0f), v1(tx, ty, 1.0f);
        if (v0.squaredNorm() > 1e-4f && v1.squaredNorm() > 1e-4f) {
            v0.normalize(); v1.normalize();
            nori::Vector3f axis = v0.cross(v1);
            float sa = std::sqrt(axis.dot(axis)),
                  ca = v0.dot(v1),
                  angle = std::atan2(sa, ca);
            if (tx*tx + ty*ty > 1.0f)
                angle *= 1.0f + 0.2f * (std::sqrt(tx*tx + ty*ty) - 1.0f);
            m_incr = Eigen::AngleAxisf(angle, axis.normalized());
            if (!std::isfinite(m_incr.norm()))
                m_incr = Quaternionf::Identity();
        }
        return true;
    }

    Eigen::Matrix4f matrix() const {
        Eigen::Matrix4f result2 = Eigen::Matrix4f::Identity();
        result2.block<3,3>(0, 0) = (m_incr * m_quat).toRotationMatrix();
        return result2;
    }


private:
    /// Whether or not this Arcball is currently active.
    bool m_active;

    /// The last click position (which triggered the Arcball to be active / non-active).
    nori::Vector2i m_lastPos;

    /// The size of this Arcball.
    nori::Vector2i m_size;

    /**
     * The current stable state.  When this Arcball is active, represents the
     * state of this Arcball when \ref Arcball::button was called with
     * ``down = true``.
     */
    Quaternionf m_quat;

    /// When active, tracks the overall update to the state.  Identity when non-active.
    Quaternionf m_incr;

    /**
     * The speed at which this Arcball rotates.  Smaller values mean it rotates
     * more slowly, higher values mean it rotates more quickly.
     */
    float m_speedFactor;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

enum TransportType : int {
    S_BLUR_INDIRECT = 0,
    S_BLUR_DIRECT,
    S_FULL,
    S_EIGENVECTORS
};

enum ColorType : int {
    C_ELI = 0,
    C_EIGENVECTOR
};

class PathGraphScreen : public Screen {
    public:

    PathGraphScreen():Screen(Vector2i(2048, 2048), "warptest: Sampling and Warping"){
        m_prefix = "";
        inc_ref();
        initializeGUI();
    }

    PathGraphScreen(std::string prefix):Screen(Vector2i(2048, 2048), "warptest: Sampling and Warping"){
        m_prefix += prefix;
        inc_ref();
        initializeGUI();
    }

    void computeTexture(std::vector<float> &pixels){
        // std::cout<<"\n computing texture "<<std::endl;
        for (int i = 0 ; i < m_pg.m_pathCount ; i++){
            if (m_pg.m_cpl[i].numOfPathPoints > 0){
                int sid = m_pg.m_cpl[i].firstPathPointIdx;
                int pid = m_pg.m_cpl[i].yIdx * m_pg.m_xresolution + m_pg.m_cpl[i].xIdx;
                if (m_pg.m_sps.m_result.iter > 0 && m_colorType == ColorType::C_ELI) {
                    if (m_transportType == TransportType::S_BLUR_INDIRECT){
                        pixels[pid * 3] = (m_pg.m_sps.m_result.blur_results[m_iter] + sid)->value[0]*m_exposure;
                        pixels[pid * 3 + 1] = (m_pg.m_sps.m_result.blur_results[m_iter] + sid)->value[1]*m_exposure;
                        pixels[pid * 3 + 2] = (m_pg.m_sps.m_result.blur_results[m_iter] + sid)->value[2]*m_exposure;
                    } else if (m_transportType == TransportType::S_BLUR_DIRECT) {
                        pixels[pid * 3] = (m_pg.m_sps.m_result.blur_direct + sid)->value[0]*m_exposure;
                        pixels[pid * 3 + 1] = (m_pg.m_sps.m_result.blur_direct + sid)->value[1]*m_exposure;
                        pixels[pid * 3 + 2] = (m_pg.m_sps.m_result.blur_direct + sid)->value[2]*m_exposure;
                    } else if (m_transportType == TransportType::S_FULL){
                        pixels[pid * 3] = ((m_pg.m_sps.m_result.mc_results[m_iter] + sid)->value[0] + m_pg.m_sps.h_sps[sid].eLd[0])*m_exposure;
                        pixels[pid * 3 + 1] = ((m_pg.m_sps.m_result.mc_results[m_iter] + sid)->value[1] +  m_pg.m_sps.h_sps[sid].eLd[1])*m_exposure;
                        pixels[pid * 3 + 2] = ((m_pg.m_sps.m_result.mc_results[m_iter] + sid)->value[2] +  m_pg.m_sps.h_sps[sid].eLd[2])*m_exposure;
                    }
                } else if (m_colorType == ColorType::C_EIGENVECTOR){
                    pixels[pid * 3] = abs(m_pg.m_eigenvector[sid]) * m_exposure;
                    pixels[pid * 3+1] = abs(m_pg.m_eigenvector[sid]) * m_exposure;
                    pixels[pid * 3+2] = abs(m_pg.m_eigenvector[sid]) * m_exposure;
                } else{
                    pixels[pid * 3] = m_pg.m_sps.h_sps[sid].eLi[0]*m_exposure;
                    pixels[pid * 3+1] = m_pg.m_sps.h_sps[sid].eLi[1]*m_exposure;
                    pixels[pid * 3+2] = m_pg.m_sps.h_sps[sid].eLi[2]*m_exposure;
                }
            }
        }
    }

    void setImage(){
        m_texture->upload((uint8_t *) m_image.data());
        m_image_view->set_image(m_texture);
        m_image_view->center();
    }

    void initializeGUI(){
        // load points to pg
        m_pg.loadGraph("/home/xd/Research/pathrenderer/scenes/" + m_prefix);

        m_window = new Window(this, "Path Graph");
        m_window->set_position(Vector2i(15, 15));
        m_window->set_layout(new GroupLayout());

        image_window = new Window(this, "Selected image");
        image_window->set_position(Vector2i(1280, 15));
        image_window->set_layout(new GroupLayout(3));
        image_window->set_visible(false);

        Widget *panel1 = new Widget(image_window);
        panel1->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 10));

        // Create a Texture instance for each object
        Vector2i size;
        int n = 0;
        
        m_image.resize(m_pg.m_xresolution * m_pg.m_yresolution * 3);
        computeTexture(m_image);
        m_texture = new Texture(Texture::PixelFormat::RGB,
                                        Texture::ComponentFormat::Float32,
                                        nanogui::Vector2i(m_pg.m_xresolution, m_pg.m_yresolution),
                                        Texture::InterpolationMode::Nearest,
                                        Texture::InterpolationMode::Nearest);

        // m_texture->upload((uint8_t *) m_image.data());

        m_image_view = new ImageView(image_window);
        m_image_view->set_size(Vector2i(1024, 1024));
        setImage();

        m_transportPhaseBox = new ComboBox(panel1, {"blur indirect", "blur direct", "full"});
        m_transportPhaseBox->set_callback([&](int){
            m_transportType = (TransportType) m_transportPhaseBox->selected_index();
            refresh();
            computeTexture(m_image);
            setImage();
        });
       

        m_image_view->set_pixel_callback(
            [this](const Vector2i& index, char **out, size_t size) {
                const Texture *texture = m_texture;
                float *dataf = new float [texture->size().x() * texture->size().y() * 3];
                m_texture->download((uint8_t *)dataf);
                // uint8_t *data = (uint8_t * )dataf;
                for (int ch = 0; ch < 3; ++ch) {
                    float value = dataf[(index.x() + index.y() * texture->size().x())*3 + ch];
                    snprintf(out[ch], size, "%.5f", (float) value);
                    // std::cout<<"pixel pick"<<value<<std::endl;
                }
            }
        );

        set_resize_callback([&](Vector2i size) {
            m_arcball.setSize(nori::Vector2i(size.x(), size.y()));
        });

        m_arcball.setSize(nori::Vector2i(m_size.x(), m_size.y()));

        m_renderPass = new RenderPass({ this });
        m_renderPass->set_clear_color(0, Color(0.f, 0.f, 0.f, 1.f));

        new Label(m_window, "Pixel Picker");

        Widget *panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

         
        m_colorTypeBox = new ComboBox(panel, {"eLi", "eigenvectors"});
        m_colorTypeBox->set_callback([&](int){
            m_colorType = (ColorType) m_colorTypeBox->selected_index();
            refresh();
        });

        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_xpixelPicker = new Slider(panel);
        m_xpixelPicker->set_fixed_width(55);
        m_xpixelPicker->set_callback([&](float) { refresh(); });

        m_ypixelPicker = new Slider(panel);
        m_ypixelPicker->set_fixed_width(55);
        m_ypixelPicker->set_callback([&](float) { refresh(); });

        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_xposText = new TextBox(panel);
        m_xposText->set_fixed_size(Vector2i(80, 25));

        m_yposText = new TextBox(panel);
        m_yposText->set_fixed_size(Vector2i(80, 25));

        new Label(m_window, "Path Graph Options");
        new Label(m_window, "K neighbor");
        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_kSlider = new Slider(panel);
        m_kSlider->set_fixed_width(150);
        m_kSlider->set_callback([&](float) { 
            m_k = floor(m_kSlider->value() * 65) + 1; 
            std::string str = tfm::format("%i", m_k);
            m_kText->set_value(str);
            });

        m_kText = new TextBox(panel);
        m_kText->set_fixed_size(Vector2i(50, 25));

        new Label(m_window, "Iterations ");
        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_iterSlider = new Slider(panel);
        m_iterSlider->set_fixed_width(150);
        m_iterSlider->set_callback([&](float) {
            m_iter = floor(m_iterSlider->value() * 80); 
            std::string str = tfm::format("%i", m_iter);
            m_iterText->set_value(str);
            computeTexture(m_image);
            setImage();
            refresh();
        });

        m_iterText = new TextBox(panel);
        m_iterText->set_fixed_size(Vector2i(50, 25));

        new Label(m_window, "Large Eigen Component");
        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_eigenSlider = new Slider(panel);
        m_eigenSlider->set_fixed_width(150);
        m_eigenSlider->set_callback([&](float) {
            m_eigen_idx = floor(m_eigenSlider->value() * 6); 
            std::string str = tfm::format("%d", m_eigen_idx);
            m_eigenIdxText->set_value(str);
            searchIdx(m_idxoffset, m_pathSegment);
            updatePath();
        });

        m_eigenIdxText = new TextBox(panel);
        m_eigenIdxText->set_fixed_size(Vector2i(50, 25));

        new Label(m_window, "Exposure ");
        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Horizontal, Alignment::Middle, 0, 20));

        m_expSlider = new Slider(panel);
        m_expSlider->set_fixed_width(150);
        m_expSlider->set_callback([&](float) {
            m_exposure = pow(2.0, m_expSlider->value() * 30.0 - 5.0); 
            std::string str = tfm::format("%.2f", m_exposure);
            m_expText->set_value(str);
            if (image_window->visible()){
                computeTexture(m_image);
                setImage();
            }
        });

        m_expText = new TextBox(panel);
        m_expText->set_fixed_size(Vector2i(70, 25));

        new Label(m_window, "GPU command");
        panel = new Widget(m_window);
        panel->set_layout(new BoxLayout(Orientation::Vertical, Alignment::Middle, 0, 20));

        Button *neighborBtn = new Button(panel, "KNN");
        neighborBtn->set_fixed_size(Vector2i(150, 25));
        neighborBtn->set_background_color(Color(0, 255, 0, 25));
        neighborBtn->set_callback([&]{
            m_pg.computeDimensions(m_pg.m_aabb, m_pg.m_sps.num);
            m_pg.m_sps.getReadyForGPU();
            // m_pg.m_sps.BuildKNN(m_k, m_k);
            m_pg.m_sps.BuildClusters(m_k);
        });

        Button *jacobianIterBtn = new Button(panel, "Jacobian Iter");
        jacobianIterBtn->set_fixed_size(Vector2i(150, 25));
        jacobianIterBtn->set_background_color(Color(0, 255, 0, 25));
        jacobianIterBtn->set_callback([&]{
            image_window->set_visible(true);
            // std::cout<<"1"<<std::endl;
            m_pg.m_sps.ClusterScatter(90);
            // std::cout<<"2"<<std::endl;
            computeTexture(m_image);
            setImage();
        });

        Button *resetBtn = new Button(panel, "Reset");
        resetBtn->set_fixed_size(Vector2i(150, 25));
        resetBtn->set_background_color(Color(0, 255, 0, 25));
        resetBtn->set_callback([&]{
            m_pg.m_sps.freeResultSpace();
            computeTexture(m_image);
            setImage();
        });

        Button *closeImageViewBtn = new Button(panel1, "Close");
        closeImageViewBtn->set_fixed_size(Vector2i(150, 25));
        closeImageViewBtn->set_background_color(Color(255, 0, 0, 25));
        closeImageViewBtn->set_callback([&]{
            image_window->set_visible(false);
        });

        perform_layout();

        m_pathShader = new Shader(
            m_renderPass,
            "Path shader",

            /* Vertex shader */
            R"(#version 330
            uniform mat4 mvp;
            in vec3 position;
            in vec3 color;
            out vec3 frag_color;
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
                if (isnan(position.r)) /* nan (missing value) */
                    frag_color = vec3(0.0);
                else
                    frag_color = color;
            })",

            /* Fragment shader */
            R"(#version 330
            in vec3 frag_color;
            out vec4 out_color;
            void main() {
                if (frag_color == vec3(0.0))
                    discard;
                out_color = vec4(frag_color, 1.0);
            })"
        );

        m_pointShader = new Shader(
            m_renderPass,
            "Point shader",

            /* Vertex shader */
            R"(#version 330
            uniform mat4 mvp;
            in vec3 position;
            in vec3 color;
            out vec3 frag_color;
            void main() {
                gl_Position = mvp * vec4(position, 1.0);
                if (isnan(position.r)) /* nan (missing value) */
                    frag_color = vec3(0.0);
                else
                    frag_color = color;
            })",

            /* Fragment shader */
            R"(#version 330
            uniform float exposure;
            in vec3 frag_color;
            out vec4 out_color;
            void main() {
                if (frag_color == vec3(0.0))
                    discard;
                out_color = vec4(frag_color*exposure, 1.0);
            })"
        );

        // initialization
        m_pointCount = 4;
        m_zoomIn = 1.0;
        m_k = 1;
        m_iter = 1;
        m_transportType=TransportType::S_BLUR_INDIRECT;
        m_exposure = 1.0;

        m_eigen_idx = 0;

        m_expSlider->set_value(0.5); 
        m_kText->set_value("1");
        m_iterText->set_value("1");
        m_expText->set_value("0.0");
        

        refresh();
        set_visible(true);
        draw_all();
    }

    void searchIdx(int &startIdx, int &number){
        int idx = m_pg.m_max_idx[m_eigen_idx];
        int i = idx;
        std::cout<<"idx: "<<idx<<std::endl; 
        
        while (i > idx - 10) {std::cout<<m_pg.m_sps.h_sps[i].nidx<<" "<<(m_pg.m_sps.h_sps[i].nidx != 0)<<std::endl;}

        startIdx = i+1;
        while (m_pg.m_sps.h_sps[i].nidx != 0) i++;
        
        number = i - startIdx;
        std::cout<<"idx: "<<startIdx<<" num: "<<number<<std::endl; 
    }

    void updatePath(){
        if (m_pathSegment > 0){
            nori::MatrixXf vertices;
            nori::MatrixXf vertexcolors;
            vertices.resize(3, m_pathSegment+1);
            vertexcolors.resize(3, m_pathSegment+1);
            vertices.col(0) = Point3f(-m_pg.m_camera_matrix.col(3)[0], -m_pg.m_camera_matrix.col(3)[1], m_pg.m_camera_matrix.col(3)[2]);
            vertexcolors.col(0) = Point3f(0.0, 1.0, 0.0);
            std::cout<<"idxoffset: "<<m_idxoffset<<"point length"<<m_pg.m_spCount<<std::endl;
            std::cout<<"startpoint: "<<vertices.col(0)<<std::endl;
            for (int i = 1 ; i <= m_pathSegment ; i++){
                int iidx = m_idxoffset+i;
                vertices.col(i) = Point3f(m_pg.m_sps.h_sps[iidx].pos[0], m_pg.m_sps.h_sps[iidx].pos[1], m_pg.m_sps.h_sps[iidx].pos[2]);
                vertexcolors.col(i) = Point3f(0.0, 1.0, 0.0);
            }
            std::cout<<"startpoint 1bounce: "<<vertices.col(1)<<std::endl;
            m_pathShader->set_buffer("position", VariableType::Float32, {(size_t) m_pathSegment+1, 3}, vertices.data());
            m_pathShader->set_buffer("color", VariableType::Float32, {(size_t) m_pathSegment+1, 3}, vertexcolors.data());
        }
    }

    

    void refresh(){
        std::cout<<"Refresh: "<<std::endl;

        m_xpos = floor(m_xpixelPicker->value() * (m_pg.m_xresolution-1));
        m_ypos = floor(m_ypixelPicker->value() * (m_pg.m_yresolution-1));

        int pathIdx = m_ypos * m_pg.m_xresolution + m_xpos;
        nori::MatrixXu indices;
        m_pathSegment = m_pg.m_cpl[pathIdx].numOfPathPoints;
        std::cout<<"pathIdx: "<<pathIdx<<" path segment "<<m_pathSegment<<std::endl;
        if (m_pathSegment > 0){
            nori::MatrixXf vertices;
            nori::MatrixXf vertexcolors;
            vertices.resize(3, m_pathSegment+1);
            vertexcolors.resize(3, m_pathSegment+1);
            m_idxoffset = m_pg.m_cpl[pathIdx].firstPathPointIdx;
            vertices.col(0) = Point3f(-m_pg.m_camera_matrix.col(3)[0], -m_pg.m_camera_matrix.col(3)[1], m_pg.m_camera_matrix.col(3)[2]);
            vertexcolors.col(0) = Point3f(0.0, 1.0, 0.0);
            std::cout<<"idxoffset: "<<m_idxoffset<<"point length"<<m_pg.m_spCount<<std::endl;
            std::cout<<"startpoint: "<<vertices.col(0)<<std::endl;
            
            std::cout<<"xpos"<<m_pg.m_cpl[pathIdx].xIdx<<" ypos:"<<m_pg.m_cpl[pathIdx].yIdx<<std::endl;
            for (int i = 1 ; i <= m_pathSegment ; i++){
                int iidx = m_idxoffset+i;
                vertices.col(i) = Point3f(m_pg.m_sps.h_sps[iidx].pos[0], m_pg.m_sps.h_sps[iidx].pos[1], m_pg.m_sps.h_sps[iidx].pos[2]);
                vertexcolors.col(i) = Point3f(0.0, 1.0, 0.0);
            }
            std::cout<<"startpoint 1bounce: "<<vertices.col(1)<<std::endl;
            
            indices.resize(2, m_pathSegment);
            
            for (int i = 0 ; i < m_pathSegment ;i++){
                indices.col(i) << i, i + 1;
            }
            m_pathShader->set_buffer("position", VariableType::Float32, {(size_t) m_pathSegment+1, 3}, vertices.data());
            m_pathShader->set_buffer("color", VariableType::Float32, {(size_t) m_pathSegment+1, 3}, vertexcolors.data());
        }

        std::cout<<"m_pathsegment="<<m_pathSegment<<std::endl;
        nori::MatrixXf positions, colors;
        positions.resize(3, m_pg.m_spCount);
        colors.resize(3, m_pg.m_spCount);
        m_pointCount = m_pg.m_spCount;
        for (int i = 0 ; i < m_pointCount ; i++){
            positions.col(i) = Point3f(m_pg.m_sps.h_sps[i].pos[0], m_pg.m_sps.h_sps[i].pos[1], m_pg.m_sps.h_sps[i].pos[2]);
            if (m_colorType == ColorType::C_ELI){
                if (m_transportType == TransportType::S_BLUR_INDIRECT && m_pg.m_sps.m_result.iter > 0){
                    colors.col(i) = Point3f((m_pg.m_sps.m_result.blur_results[m_iter] + i)->value[0], 
                                            (m_pg.m_sps.m_result.blur_results[m_iter] + i)->value[1], 
                                            (m_pg.m_sps.m_result.blur_results[m_iter] + i)->value[2]);
                } else if(m_transportType == TransportType::S_BLUR_DIRECT && m_pg.m_sps.m_result.iter > 0){
                    colors.col(i) = Point3f((m_pg.m_sps.m_result.blur_direct + i)->value[0], 
                                            (m_pg.m_sps.m_result.blur_direct + i)->value[1], 
                                            (m_pg.m_sps.m_result.blur_direct + i)->value[2]);
                } else if(m_transportType == TransportType::S_FULL && m_pg.m_sps.m_result.iter > 0){
                    colors.col(i) = Point3f((m_pg.m_sps.m_result.mc_results[m_iter] + i)->value[0], 
                                            (m_pg.m_sps.m_result.mc_results[m_iter] + i)->value[1], 
                                            (m_pg.m_sps.m_result.mc_results[m_iter] + i)->value[2]);
                } else {
                    colors.col(i) = Point3f(m_pg.m_sps.h_sps[i].eLi[0], m_pg.m_sps.h_sps[i].eLi[1], m_pg.m_sps.h_sps[i].eLi[2]);
                }
            }
            else if (m_colorType == ColorType::C_EIGENVECTOR){
                colors.col(i) = Point3f(abs(m_pg.m_eigenvector[i]), abs(m_pg.m_eigenvector[i]), abs(m_pg.m_eigenvector[i]));
            }
        }

        m_sceneRadius = m_pg.m_aabb.extents[m_pg.m_aabb.longAxis];
        m_sceneCenter = Vector3f(m_pg.m_aabb.center[0], m_pg.m_aabb.center[1], m_pg.m_aabb.center[2]);
        m_pointShader->set_buffer("position", VariableType::Float32, {(size_t) m_pointCount, 3}, positions.data());
        m_pointShader->set_buffer("color", VariableType::Float32, {(size_t) m_pointCount, 3}, colors.data());
    
        std::string str;
        m_xposText->set_units(" ");
        str = tfm::format("%i", m_xpos);
        m_xposText->set_value(str);

        m_yposText->set_units(" ");
        str = tfm::format("%i", m_ypos);
        m_yposText->set_value(str);
    }

    void ETON(Matrix4f &a, const Eigen::Matrix4f &b){
        a.m[0][0] = b(0, 0);
        a.m[0][1] = b(0, 1);
        a.m[0][2] = b(0, 2);
        a.m[0][3] = b(0, 3);

        a.m[1][0] = b(1, 0);
        a.m[1][1] = b(1, 1);
        a.m[1][2] = b(1, 2);
        a.m[1][3] = b(1, 3);

        a.m[2][0] = b(2, 0);
        a.m[2][1] = b(2, 1);
        a.m[2][2] = b(2, 2);
        a.m[2][3] = b(2, 3);

        a.m[3][0] = b(3, 0);
        a.m[3][1] = b(3, 1);
        a.m[3][2] = b(3, 2);
        a.m[3][3] = b(3, 3);
    }

    void draw_contents(){
        Matrix4f view, proj, model(1.f);
        
        // ETON(view, m_pg.m_camera_matrix);
        // view = Matrix4f::rotate(Vector3f(0.0, 1.0, 0.0), 180) * view;
        // view = Matrix4f::translate(Vector3f(0.0, 0.0, m_sceneRadius * m_zoomIn)) * view;
        view = Matrix4f::look_at(Vector3f(m_sceneCenter) - Vector3f(0.0, 0.0, 1.5 * m_sceneRadius + 0.25 * m_zoomIn), Vector3f(m_sceneCenter) - Vector3f(0.0, 0.0, m_sceneRadius), Vector3f(0, 1, 0));
        const float viewAngle = m_pg.m_fov, near_clip = m_pg.m_near_clip, far_clip = 100000;
        // std::cout<<"ViewAngle: "<<viewAngle<<std::endl;
        // ETON(proj, m_pg.m_camera2sample);
        proj = Matrix4f::perspective(viewAngle / 180.0f * M_PI, near_clip, far_clip,
                                    (float) m_size.x() / (float) m_size.y());
        // model = Matrix4f::translate(Vector3f(-0.5f, -0.5f, 0.0f)) * model;

        Matrix4f arcball_ng(1.f);
        memcpy(arcball_ng.m, m_arcball.matrix().data(), sizeof(float) * 16);
        model = arcball_ng * model;
        //framebuffer_size();
        m_renderPass->resize(m_size);
        m_renderPass->begin();
        
        /* Render the point set */
        Matrix4f mvp = proj * view * model;
        m_pointShader->set_uniform("mvp", mvp);
        m_pointShader->set_uniform("exposure", m_exposure);
        glPointSize(4);
        m_renderPass->set_depth_test(RenderPass::DepthTest::Less, true);
        m_pointShader->begin();
        m_pointShader->draw_array(nanogui::Shader::PrimitiveType::Point, 0, m_pointCount);
        m_pointShader->end();

        if(m_pathSegment > 0){
            m_pathShader->set_uniform("mvp", mvp);
            glPointSize(4);
            glLineWidth(4);
            m_pathShader->begin();
            m_pathShader->draw_array(nanogui::Shader::PrimitiveType::LineStrip, 0, m_pathSegment);
            m_pathShader->end();
        }

        m_renderPass->end();
    }

    bool mouse_motion_event(const Vector2i &p, const Vector2i &rel,
                                  int button, int modifiers) {
        if (!Screen::mouse_motion_event(p, rel, button, modifiers))
            m_arcball.motion(nori::Vector2i(p.x(), p.y()));
        return true;
    }

    bool mouse_button_event(const Vector2i &p, int button, bool down, int modifiers) {
        if (down && !m_window->visible()) {
            m_window->set_visible(true);
            return true;
        }
        
        if (!Screen::mouse_button_event(p, button, down, modifiers)) {
            if (button == GLFW_MOUSE_BUTTON_1){
                // std::cout<<"arcball: btn="<<button<<" GLFW_MOUSE_BUTTON_1= "<<GLFW_MOUSE_BUTTON_1<<std::endl;
                m_arcball.button(nori::Vector2i(p.x(), p.y()), down);
            }
        }
        return true;
    }

    bool scroll_event(const Vector2i &p, const Vector2f &rel){
        if (!Screen::scroll_event(p, rel)){
            // std::cout<<"scroll p = "<<p.x()<<" "<<p.y() << " rel = "<<rel<<std::endl;
            m_zoomIn = rel.y() * 0.1 * m_sceneRadius + m_zoomIn;
        }
        return true;
    }

    private:
    Arcball m_arcball;
    Window *m_window, *image_window;
    PathGraph m_pg;
    nanogui::ref<Shader> m_pointShader;
    nanogui::ref<Shader> m_pathShader;
    nanogui::ref<RenderPass> m_renderPass;
    size_t m_pointCount;
    float m_sceneRadius;
    Vector3f m_sceneCenter;
    float m_zoomIn;

    ComboBox *m_transportPhaseBox;
    ComboBox *m_colorTypeBox;

    TransportType m_transportType;
    ColorType m_colorType;
    
    // Slider
    Slider *m_xpixelPicker;
    Slider *m_ypixelPicker;
    Slider *m_kSlider;
    Slider *m_dkSlider;
    Slider *m_iterSlider;
    Slider *m_expSlider;
    Slider *m_eigenSlider;


    TextBox *m_xposText;
    TextBox *m_yposText;
    TextBox *m_kText;
    TextBox *m_iterText;
    TextBox *m_expText;
    TextBox *m_eigenIdxText;

    int m_xpos;
    int m_ypos;
    int m_pathSegment;
    int m_k;
    int m_dk;
    int m_iter;
    int m_eigen_idx;
    int m_idxoffset;

    float m_exposure;

    std::vector<float> m_image;
    Texture * m_texture;
    ImageView *m_image_view;

    std::string m_prefix = "";
};

int main(int argc, char **argv){
    std::cout<<"argc:"<<argc<<"argv: "<<argv[1];
    if (argc <= 1) {
        std::cout<<"argc less or equal to 1"<<std::endl;
        nanogui::init();
        PathGraphScreen *graph = new PathGraphScreen();
        nanogui::mainloop();
        delete graph;
        nanogui::shutdown();
        return 0;
    } else {
        std::cout<<"argc larger than 1"<<std::endl;
        nanogui::init();
        PathGraphScreen *graph = new PathGraphScreen(std::string(argv[1]));
        nanogui::mainloop();
        delete graph;
        nanogui::shutdown();
        return 0;
    }
    return 0;
}