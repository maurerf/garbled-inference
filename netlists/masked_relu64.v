//file: relu.v
`timescale 1ns / 1ps
module relu(
    input [63:0] g_input, //mask
    input [63:0] e_input, //relu input
    output [63:0] o
    );
    wire [63:0] unmasked;
    assign unmasked = g_input + e_input; // "unmask" input
    assign o = (unmasked[63] == 0)? unmasked + g_input : g_input;   //if the sign bit is high, send masked zero on the output else send the un- and then remasked input
endmodule
