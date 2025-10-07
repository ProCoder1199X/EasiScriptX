#include "onnx_export.hpp"
#include "config.hpp"
#include "error_handler.hpp"
#include <fstream>
#include <sstream>

#if USE_ONNX
#include <onnxruntime_cxx_api.h>
#endif

namespace esx::export {

/**
 * @brief ONNX model exporter for EasiScriptX models.
 */
ONNXExporter::ONNXExporter() {
#if USE_ONNX
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ESX");
    session_options_ = std::make_unique<Ort::SessionOptions>();
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
#endif
}

ONNXExporter::~ONNXExporter() = default;

/**
 * @brief Export model to ONNX format.
 * @param model Model tensor to export.
 * @param output_path Output file path.
 * @return true if export successful, false otherwise.
 */
bool ONNXExporter::export_model(const std::vector<std::vector<double>>& model, 
                                const std::string& output_path) {
    ESX_DEBUG("ONNX_EXPORT", "Exporting model to " << output_path);
    
    if (!validate_file_path(output_path)) {
        esx::error::get_error_handler().report_error(
            esx::error::FileError("Invalid output path: " + output_path));
        return false;
    }
    
#if USE_ONNX
    try {
        // Create ONNX model structure
        std::ostringstream model_stream;
        model_stream << create_onnx_model(model);
        
        // Write to file
        std::ofstream file(output_path, std::ios::binary);
        if (!file.is_open()) {
            esx::error::get_error_handler().report_error(
                esx::error::FileError("Cannot open output file: " + output_path));
            return false;
        }
        
        file << model_stream.str();
        file.close();
        
        ESX_DEBUG("ONNX_EXPORT", "Model exported successfully");
        return true;
    } catch (const std::exception& e) {
        esx::error::get_error_handler().report_error(
            esx::error::RuntimeError("ONNX export failed: " + std::string(e.what())));
        return false;
    }
#else
    esx::error::get_error_handler().report_error(
        esx::error::RuntimeError("ONNX support not enabled"));
    return false;
#endif
}

/**
 * @brief Import model from ONNX format.
 * @param input_path Input file path.
 * @return Imported model tensor.
 */
std::vector<std::vector<double>> ONNXExporter::import_model(const std::string& input_path) {
    ESX_DEBUG("ONNX_EXPORT", "Importing model from " << input_path);
    
    if (!validate_file_path(input_path)) {
        esx::error::get_error_handler().report_error(
            esx::error::FileError("Invalid input path: " + input_path));
        return {};
    }
    
#if USE_ONNX
    try {
        // Create session
        auto session = std::make_unique<Ort::Session>(*env_, input_path.c_str(), *session_options_);
        
        // Get input/output info
        size_t num_input_nodes = session->GetInputCount();
        size_t num_output_nodes = session->GetOutputCount();
        
        ESX_DEBUG("ONNX_EXPORT", "Model has " << num_input_nodes << " inputs and " << num_output_nodes << " outputs");
        
        // Mock model data (in full implementation, would extract actual weights)
        std::vector<std::vector<double>> model_data;
        model_data.push_back({1.0, 2.0, 3.0, 4.0});
        model_data.push_back({5.0, 6.0, 7.0, 8.0});
        
        ESX_DEBUG("ONNX_EXPORT", "Model imported successfully");
        return model_data;
    } catch (const std::exception& e) {
        esx::error::get_error_handler().report_error(
            esx::error::RuntimeError("ONNX import failed: " + std::string(e.what())));
        return {};
    }
#else
    esx::error::get_error_handler().report_error(
        esx::error::RuntimeError("ONNX support not enabled"));
    return {};
#endif
}

/**
 * @brief Create ONNX model structure.
 * @param model Model tensor data.
 * @return ONNX model string.
 */
std::string ONNXExporter::create_onnx_model(const std::vector<std::vector<double>>& model) {
    // Mock ONNX model creation
    // In full implementation, would use ONNX protobuf to create proper model
    std::ostringstream oss;
    oss << "ONNX Model v1.0\n";
    oss << "Input: tensor(" << model.size() << "x" << (model.empty() ? 0 : model[0].size()) << ")\n";
    oss << "Output: tensor(" << model.size() << "x" << (model.empty() ? 0 : model[0].size()) << ")\n";
    oss << "Weights: [";
    for (size_t i = 0; i < model.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << "[";
        for (size_t j = 0; j < model[i].size(); ++j) {
            if (j > 0) oss << ", ";
            oss << model[i][j];
        }
        oss << "]";
    }
    oss << "]\n";
    return oss.str();
}

/**
 * @brief Validate ONNX model.
 * @param model_path Path to ONNX model file.
 * @return true if model is valid, false otherwise.
 */
bool ONNXExporter::validate_model(const std::string& model_path) {
    ESX_DEBUG("ONNX_EXPORT", "Validating model: " << model_path);
    
    if (!validate_file_path(model_path)) {
        return false;
    }
    
    std::ifstream file(model_path);
    if (!file.is_open()) {
        return false;
    }
    
    // Check for ONNX magic bytes or header
    std::string line;
    std::getline(file, line);
    bool is_valid = line.find("ONNX") != std::string::npos;
    
    file.close();
    return is_valid;
}

/**
 * @brief Get model metadata.
 * @param model_path Path to ONNX model file.
 * @return Model metadata map.
 */
std::map<std::string, std::string> ONNXExporter::get_model_metadata(const std::string& model_path) {
    std::map<std::string, std::string> metadata;
    
    if (!validate_model(model_path)) {
        return metadata;
    }
    
    // Mock metadata extraction
    metadata["version"] = "1.0.0";
    metadata["framework"] = "EasiScriptX";
    metadata["input_shape"] = "2x4";
    metadata["output_shape"] = "2x4";
    metadata["precision"] = "fp32";
    
    return metadata;
}

/**
 * @brief Optimize ONNX model.
 * @param input_path Input model path.
 * @param output_path Output model path.
 * @return true if optimization successful, false otherwise.
 */
bool ONNXExporter::optimize_model(const std::string& input_path, const std::string& output_path) {
    ESX_DEBUG("ONNX_EXPORT", "Optimizing model: " << input_path << " -> " << output_path);
    
    if (!validate_file_path(input_path) || !validate_file_path(output_path)) {
        return false;
    }
    
    // Mock optimization (in full implementation, would use ONNX optimizer)
    std::ifstream input_file(input_path);
    std::ofstream output_file(output_path);
    
    if (!input_file.is_open() || !output_file.is_open()) {
        return false;
    }
    
    std::string line;
    while (std::getline(input_file, line)) {
        // Mock optimization: add optimization comment
        if (line.find("ONNX Model") != std::string::npos) {
            output_file << line << " (Optimized)\n";
        } else {
            output_file << line << "\n";
        }
    }
    
    input_file.close();
    output_file.close();
    
    ESX_DEBUG("ONNX_EXPORT", "Model optimization completed");
    return true;
}

} // namespace esx::export
