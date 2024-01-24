/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2023 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/float_equal.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/quantize_fp16.hpp>
#include <migraphx/program.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/iterator_for.hpp>
#include <migraphx/stringutils.hpp>
#include <migraphx/ranges.hpp>
#include <migraphx/target.hpp>
#include <migraphx/make_op.hpp>

namespace migraphx {
inline namespace MIGRAPHX_INLINE_NS {

static void quantize_module(module& m, const std::vector<std::string>& ins_names)
{
    for(auto ins : iterator_for(m))
    {
        // instructions are not in the set to be quantized
        if(not(contains(ins_names, ins->name()) or contains(ins_names, "all")))
            continue;

        // skip return and convert instructions
        if(contains({"@return", "convert"}, ins->name()))
            continue;

        if(ins->inputs().empty())
            continue;

        auto mod_inputs = ins->module_inputs();
        auto s          = ins->get_shape();
        // Convert each of the inputs that are floating point to fp16
        auto inputs = ins->inputs();
        std::transform(inputs.begin(), inputs.end(), inputs.begin(), [&](auto input) {
            auto input_type = input->get_shape().type();
            if(s.type() == shape::tuple_type)
            {
                std::cout << "## Input shape " << input->get_shape() << std::endl;
            }
            if(input_type != shape::float_type and input_type != shape::double_type)
                return input;
            return m.insert_instruction(
                ins, make_op("convert", {{"target_type", shape::half_type}}), input);
        });

        // Insert quantized ins
        auto converted_ins = m.insert_instruction(ins, ins->get_operator(), inputs, mod_inputs);

        // Convert back to original type after quantizing
        if(mod_inputs.empty())
        {
            // Unpack tuple type and convert each instruct
            if(s.type() == shape::tuple_type)
            {
                std::cout << "### Unpacking touple, shape: " << s << " converted shape "
                          << converted_ins->get_shape() << std::endl;
                auto orig_outputs = ins->outputs();
                std::unordered_map<std::string, shape::type_t> original_types;
                std::transform(orig_outputs.begin(),
                               orig_outputs.end(),
                               std::inserter(original_types, original_types.end()),
                               [](const auto& instr) {
                                   return std::make_pair(instr->name(), instr->get_shape().type());
                               });

                auto outputs = converted_ins->outputs();
                std::cout << "### Outputs size " << outputs.size() << std::endl;
                std::for_each(outputs.begin(), outputs.end(), [&](auto& instr) {
                    auto orig_type = original_types.at(instr->name());
                    std::cout << "### Converting back from" << instr->get_shape().type() << " to "
                                << orig_type << std::endl;
                    if(orig_type != instr->get_shape().type())
                    {
                        converted_ins = m.insert_instruction(
                            instr, make_op("convert", {{"target_type", orig_type}}), converted_ins);
                    }
                });
            }
            else
            {
                converted_ins = m.insert_instruction(
                    ins, make_op("convert", {{"target_type", s.type()}}), converted_ins);
            }
        }
        // Replace original instruction
        m.replace_instruction(ins, converted_ins);
    }
}

void quantize_fp16_pass::apply(module& m) const { quantize_module(m, ins_names); }

} // namespace MIGRAPHX_INLINE_NS
} // namespace migraphx
