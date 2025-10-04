#ifndef ESX_ONNX_EXPORT_HPP
#define ESX_ONNX_EXPORT_HPP

#include "config.hpp"
#include <vector>
#include <string>
#include <memory>
#include <map>

#if USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

/**
 * @file onnx_export.hpp
 * @brief ONNX model export/import for EasiScriptX (ESX).
 * @details Provides functionality to export ESX models to ONNX format
 * and import ONNX models for use in ESX workflows.
 */

namespace esx::export {

/**
 * @brief ONNX model exporter and importer.
 */
class ONNXExporter {
public:
    ONNXExporter();
    ~ONNXExporter();

    /**
     * @brief Export model to ONNX format.
     * @param model Model tensor to export.
     * @param output_path Output file path.
     * @return true if export successful, false otherwise.
     */
    bool export_model(const std::vector<std::vector<double>>& model, 
                     const std::string& output_path);

    /**
     * @brief Import model from ONNX format.
     * @param input_path Input file path.
     * @return Imported model tensor.
     */
    std::vector<std::vector<double>> import_model(const std::string& input_path);

    /**
     * @brief Validate ONNX model.
     * @param model_path Path to ONNX model file.
     * @return true if model is valid, false otherwise.
     */
    bool validate_model(const std::string& model_path);

    /**
     * @brief Get model metadata.
     * @param model_path Path to ONNX model file.
     * @return Model metadata map.
     */
    std::map<std::string, std::string> get_model_metadata(const std::string& model_path);

    /**
     * @brief Optimize ONNX model.
     * @param input_path Input model path.
     * @param output_path Output model path.
     * @return true if optimization successful, false otherwise.
     */
    bool optimize_model(const std::string& input_path, const std::string& output_path);

private:
#if USE_ONNX
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
#endif

    /**
     * @brief Create ONNX model structure.
     * @param model Model tensor data.
     * @return ONNX model string.
     */
    std::string create_onnx_model(const std::vector<std::vector<double>>& model);
};

} // namespace esx::export

#endif // ESX_ONNX_EXPORT_HPP
