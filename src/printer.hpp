
#ifndef ESX_PRINTER_HPP
#define ESX_PRINTER_HPP

#include "ast.hpp"
#include <iostream>

namespace esx::ast {
struct Printer {
    void operator()(const TensorLit& t) const {
        std::cout << "tensor([";
        for (size_t i = 0; i < t.data.size(); ++i) {
            std::cout << "[";
            for (size_t j = 0; j < t.data[i].size(); ++j) {
                std::cout << t.data[i][j];
                if (j < t.data[i].size() - 1) std::cout << ",";
            }
            std::cout << "]";
            if (i < t.data.size() - 1) std::cout << ",";
        }
        std::cout << "])";
    }

    void operator()(const PipeExpr& p) const {
        std::cout << "(";
        std::visit(*this, p.lhs->get());
        std::cout << " |> " << p.op << ")";
    }

    void operator()(const TrainStmt& t) const {
        std::cout << "train(" << t.model << ", " << t.data << ", loss: " << t.loss
                  << ", opt: " << t.opt << ", params: [";
        for (size_t i = 0; i < t.opt_params.size(); ++i) {
            std::cout << t.opt_params[i].first << "=" << t.opt_params[i].second;
            if (i < t.opt_params.size() - 1) std::cout << ",";
        }
        std::cout << "], epochs=" << t.epochs << ", device: " << t.device << ")";
    }

    void operator()(const AutonomicStmt& a) const {
        std::cout << "with autonomic { ";
        std::visit(*this, a.body->get());
        std::cout << " }";
    }

    void operator()(const AgentTuneStmt& a) const {
        std::cout << "multi_agent_tune(" << a.fn << ", agents: " << a.agents 
                  << ", target: " << a.target << ")";
    }

    void operator()(const MatmulExpr& m) const {
        std::cout << "(";
        std::visit(*this, m.lhs->get());
        std::cout << " @ ";
        std::visit(*this, m.rhs->get());
        std::cout << ")";
    }

    void operator()(const IfStmt& i) const {
        std::cout << "if ";
        std::visit(*this, i.cond->get());
        std::cout << " { ";
        std::visit(*this, i.then_body->get());
        std::cout << " }";
        if (i.else_body) {
            std::cout << " else { ";
            std::visit(*this, i.else_body->get());
            std::cout << " }";
        }
    }

    void operator()(const ForStmt& f) const {
        std::cout << "for " << f.var << " in ";
        std::visit(*this, f.start->get());
        std::cout << ".." << f.end->get() << " { ";
        std::visit(*this, f.body->get());
        std::cout << " }";
    }

    void operator()(const WhileStmt& w) const {
        std::cout << "while ";
        std::visit(*this, w.cond->get());
        std::cout << " { ";
        std::visit(*this, w.body->get());
        std::cout << " }";
    }

    void operator()(const GradExpr& g) const {
        std::cout << "grad(";
        std::visit(*this, g.fn->get());
        std::cout << ")";
    }

    void operator()(const OdeSolveExpr& o) const {
        std::cout << "ode_solve(" << o.eq << ", " << o.y0 << ", [";
        for (size_t i = 0; i < o.t.size(); ++i) {
            std::cout << o.t[i];
            if (i < o.t.size() - 1) std::cout << ",";
        }
        std::cout << "])";
    }

    void operator()(const QuantizeExpr& q) const {
        std::cout << "quantize(bits=" << q.bits << ", aware: " << q.aware << ")";
    }

    void operator()(const PruneExpr& p) const {
        std::cout << "prune(ratio=" << p.ratio << ")";
    }

    void operator()(const DeployExpr& d) const {
        std::cout << "deploy(target: " << d.target << ", device: " << d.device << ")";
    }

    void operator()(const AttentionExpr& a) const {
        std::cout << "attention(";
        std::visit(*this, a.q->get());
        std::cout << ", ";
        std::visit(*this, a.k->get());
        std::cout << ", ";
        std::visit(*this, a.v->get());
        std::cout << ", heads=" << a.heads << ", dim=" << a.dim << ")";
    }

    void operator()(const GnnConvExpr& g) const {
        std::cout << "gnn_conv(";
        std::visit(*this, g.graph->get());
        std::cout << ", ";
        std::visit(*this, g.feats->get());
        std::cout << ")";
    }

    void operator()(const SqlTensorExpr& s) const {
        std::cout << "sql_tensor(";
        std::visit(*this, s.mat->get());
        std::cout << ", " << s.query << ")";
    }

    void operator()(const FactorizeExpr& f) const {
        std::cout << "factorize(";
        std::visit(*this, f.layer->get());
        std::cout << ", mode: " << f.mode << ")";
    }

    void operator()(const ScaleExpr& s) const {
        std::cout << "scale(";
        std::visit(*this, s.model->get());
        std::cout << ", factor=" << s.factor << ")";
    }

    void operator()(const RateReduceExpr& r) const {
        std::cout << "rate_reduce(";
        std::visit(*this, r.data->get());
        std::cout << ", freq=" << r.freq << ")";
    }

    void operator()(const PicoBenchExpr& p) const {
        std::cout << "pico_bench(";
        std::visit(*this, p.model->get());
        std::cout << ", suite: " << p.suite << ")";
    }

    void operator()(const MobiPruneExpr& m) const {
        std::cout << "mobi_prune(battery: " << (m.battery ? "true" : "false") << ")";
    }

    void operator()(const ReasoningPassExpr& r) const {
        std::cout << "reasoning_pass(";
        std::visit(*this, r.fn->get());
        std::cout << ", target: " << r.target << ")";
    }

    void operator()(const FnDecl& f) const {
        std::cout << "fn " << f.name << "(";
        for (size_t i = 0; i < f.params.size(); ++i) {
            std::cout << f.params[i];
            if (i < f.params.size() - 1) std::cout << ",";
        }
        std::cout << ") -> " << f.ret_type << " { ";
        std::visit(*this, f.body->get());
        std::cout << " }";
    }

    void operator()(const ModelExpr& m) const {
        std::cout << "model " << m.name << " { ";
        for (const auto& layer : m.layers) {
            std::visit(*this, layer->get());
            std::cout << "; ";
        }
        std::cout << " }";
    }

    void operator()(const DatasetExpr& d) const {
        std::cout << "load_dataset(" << d.name << ", preprocess: " << d.preprocess_fn 
                  << ", augment: [";
        for (size_t i = 0; i < d.augment_ops.size(); ++i) {
            std::cout << d.augment_ops[i];
            if (i < d.augment_ops.size() - 1) std::cout << ",";
        }
        std::cout << "])";
    }

    void operator()(const Conv2dExpr& c) const {
        std::cout << "conv2d(";
        std::visit(*this, c.input->get());
        std::cout << ", ";
        std::visit(*this, c.kernel->get());
        std::cout << ", stride=" << c.stride << ", padding=" << c.padding << ")";
    }

    void operator()(const MaxPoolExpr& m) const {
        std::cout << "maxpool(";
        std::visit(*this, m.input->get());
        std::cout << ", pool_size=(" << m.pool_size[0] << "," << m.pool_size[1] << "))";
    }

    void operator()(const BatchNormExpr& b) const {
        std::cout << "batchnorm(";
        std::visit(*this, b.input->get());
        std::cout << ")";
    }

    void operator()(const LayerNormExpr& l) const {
        std::cout << "layernorm(";
        std::visit(*this, l.input->get());
        std::cout << ")";
    }

    void operator()(const SparseTensorLit& s) const {
        std::cout << "sparse_tensor([";
        for (size_t i = 0; i < s.indices_values.size(); ++i) {
            std::cout << "[";
            for (size_t j = 0; j < s.indices_values[i].first.size(); ++j) {
                std::cout << s.indices_values[i].first[j];
                if (j < s.indices_values[i].first.size() - 1) std::cout << ",";
            }
            std::cout << "]:" << s.indices_values[i].second;
            if (i < s.indices_values.size() - 1) std::cout << ",";
        }
        std::cout << "], shape=[";
        for (size_t i = 0; i < s.shape.size(); ++i) {
            std::cout << s.shape[i];
            if (i < s.shape.size() - 1) std::cout << ",";
        }
        std::cout << "])";
    }

    void operator()(const ImportExpr& i) const {
        std::cout << "load_pretrained(" << i.name << ")";
    }

    void operator()(const ProfileStmt& p) const {
        std::cout << "profile { ";
        std::visit(*this, p.body->get());
        std::cout << " }";
    }

    void operator()(const VisualizeExpr& v) const {
        std::cout << "visualize(";
        std::visit(*this, v.data->get());
        std::cout << ", type: " << v.type << ")";
    }

    void operator()(const TokenizeExpr& t) const {
        std::cout << "tokenize(" << t.text << ", vocab: " << t.vocab << ")";
    }

    void operator()(const Expr& e) const {
        std::visit(*this, e);
    }

    void operator()(const Stmt& s) const {
        std::visit(*this, s);
    }

    void operator()(const Decl& d) const {
        std::cout << "let " << d.var << " = ";
        std::visit(*this, d.expr);
    }

    void operator()(const Program& p) const {
        for (const auto& decl : p.decls) {
            (*this)(decl);
            std::cout << ";\n";
        }
        for (const auto& stmt : p.stmts) {
            std::visit(*this, stmt);
            std::cout << ";\n";
        }
    }
};

} // namespace esx::ast

#endif // ESX_PRINTER_HPP
