
#ifndef ESX_PRINTER_HPP
#define ESX_PRINTER_HPP

#include "ast.hpp"
#include <iostream>
#include <string>

/**
 * @file printer.hpp
 * @brief AST printer for EasiScriptX (ESX).
 * @details Prints AST nodes for debugging, supporting all ESX syntax elements including
 * LLM fine-tuning, mixed-precision, and enterprise features.
 */

namespace esx::printer {

class Printer {
public:
    void print(const ast::Program& program) {
        for (const auto& stmt : program.stmts) {
            print_stmt(stmt, 0);
        }
    }

private:
    void indent(int level) {
        std::cout << std::string(level * 2, ' ');
    }

    void print_expr(const std::shared_ptr<ast::Expr>& expr, int level) {
        indent(level);
        if (auto* ident = dynamic_cast<ast::IdentExpr*>(expr.get())) {
            std::cout << "IdentExpr(name=" << ident->name << ") at line " << ident->loc.line << std::endl;
        } else if (auto* lit = dynamic_cast<ast::TensorLitExpr*>(expr.get())) {
            std::cout << "TensorLitExpr(values=...) at line " << lit->loc.line << std::endl;
        } else if (auto* matmul = dynamic_cast<ast::MatmulExpr*>(expr.get())) {
            std::cout << "MatmulExpr at line " << matmul->loc.line << std::endl;
            print_expr(matmul->left, level + 1);
            print_expr(matmul->right, level + 1);
        } else if (auto* conv = dynamic_cast<ast::Conv2dExpr*>(expr.get())) {
            std::cout << "Conv2dExpr(stride=" << conv->stride << ", padding=" << conv->padding
                      << ") at line " << conv->loc.line << std::endl;
            print_expr(conv->input, level + 1);
            print_expr(conv->kernel, level + 1);
        } else if (auto* attn = dynamic_cast<ast::AttentionExpr*>(expr.get())) {
            std::cout << "AttentionExpr(heads=" << attn->heads << ", dim=" << attn->dim
                      << ", flash=" << (attn->use_flash_attention ? "true" : "false")
                      << ") at line " << attn->loc.line << std::endl;
            print_expr(attn->q, level + 1);
            print_expr(attn->k, level + 1);
            print_expr(attn->v, level + 1);
        } else if (auto* lora = dynamic_cast<ast::LoRAExpr*>(expr.get())) {
            std::cout << "LoRAExpr(rank=" << lora->rank << ") at line " << lora->loc.line << std::endl;
            print_expr(lora->model, level + 1);
        } else if (auto* mp = dynamic_cast<ast::MixedPrecisionExpr*>(expr.get())) {
            std::cout << "MixedPrecisionExpr(precision=" << mp->precision << ") at line "
                      << mp->loc.line << std::endl;
            print_expr(mp->model, level + 1);
        } else {
            std::cout << "UnknownExpr at line " << expr->loc.line << std::endl;
        }
    }

    void print_stmt(const std::shared_ptr<ast::Stmt>& stmt, int level) {
        indent(level);
        if (auto* let = dynamic_cast<ast::LetStmt*>(stmt.get())) {
            std::cout << "LetStmt(name=" << let->name << ") at line " << let->loc.line << std::endl;
            print_expr(let->expr, level + 1);
        } else if (auto* train = dynamic_cast<ast::TrainStmt*>(stmt.get())) {
            std::cout << "TrainStmt(loss=" << train->loss << ", opt=" << train->opt
                      << ", epochs=" << train->epochs << ", device=" << train->device
                      << ") at line " << train->loc.line << std::endl;
            print_expr(train->model, level + 1);
            print_expr(train->dataset, level + 1);
        } else if (auto* pp = dynamic_cast<ast::PipelineParallelStmt*>(stmt.get())) {
            std::cout << "PipelineParallelStmt(stages=" << pp->stages << ") at line "
                      << pp->loc.line << std::endl;
            print_expr(pp->model, level + 1);
        } else if (auto* it = dynamic_cast<ast::InstructionTuneStmt*>(stmt.get())) {
            std::cout << "InstructionTuneStmt(prompts=" << it->prompts << ") at line "
                      << it->loc.line << std::endl;
            print_expr(it->model, level + 1);
            print_expr(it->dataset, level + 1);
        } else if (auto* da = dynamic_cast<ast::DomainAdaptStmt*>(stmt.get())) {
            std::cout << "DomainAdaptStmt(domain=" << da->domain << ") at line "
                      << da->loc.line << std::endl;
            print_expr(da->model, level + 1);
        } else if (auto* hs = dynamic_cast<ast::HeterogeneousScheduleStmt*>(stmt.get())) {
            std::cout << "HeterogeneousScheduleStmt(cpu=" << hs->cpu_ratio
                      << ", gpu=" << hs->gpu_ratio << ") at line " << hs->loc.line << std::endl;
        } else if (auto* ea = dynamic_cast<ast::EnergyAwareStmt*>(stmt.get())) {
            std::cout << "EnergyAwareStmt(max_power=" << ea->max_power << "W) at line "
                      << ea->loc.line << std::endl;
            print_expr(ea->model, level + 1);
        } else if (auto* sf = dynamic_cast<ast::SwitchFrameworkStmt*>(stmt.get())) {
            std::cout << "SwitchFrameworkStmt(framework=" << sf->framework
                      << ", path=" << sf->model_path << ") at line " << sf->loc.line << std::endl;
        } else if (auto* te = dynamic_cast<ast::TrackExperimentStmt*>(stmt.get())) {
            std::cout << "TrackExperimentStmt(tracker=" << te->tracker
                      << ", run_id=" << te->run_id << ") at line " << te->loc.line << std::endl;
        } else {
            std::cout << "UnknownStmt at line " << stmt->loc.line << std::endl;
        }
    }
};

} // namespace esx::printer

#endif // ESX_PRINTER_HPP
