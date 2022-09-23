//file: relu.v
`timescale 1ns / 1ps
module relu(
    input [127:0] g_input, //mask
    input [63:0] e_input, //masked step input
    output [63:0] o
    );
    wire [63:0] mask1;
    wire [63:0] mask2;
    wire [63:0] unmasked;
    assign mask1 = g_input[63:0];
    assign mask2= g_input[127:64];
    assign unmasked = e_input + mask1; // "unmask" input
    assign o = (unmasked[63] == 0)? mask2 + 1 : mask2;   //if the sign bit is high, send masked zero on the output else send masked one
endmodule
