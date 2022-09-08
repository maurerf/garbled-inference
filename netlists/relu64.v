//file: relu.v
`timescale 1ns / 1ps
module relu(
    input [63:0] g_input, //mask
    input [63:0] e_input, //relu input
    output [63:0] o
    );
    assign o = (e_input[63] == 0)? e_input : 0;   //if the sign bit is high, send masked zero on the output else send the masked input
endmodule
