from sympy import *
from sympy.printing.mathml import print_mathml

def output_latex(expression):
    result = latex(expression)
    print(result)
    return result

(R_k1, G_k1, B_k1,
 R_k2, G_k2, B_k2,
 R_f1, G_f1, B_f1,
 R_f2, G_f2, B_f2) = symbols("R_k1 G_k1 B_k1 "
                             "R_k2 G_k2 B_k2 "
                             "R_f1 G_f1 B_f1 "
                             "R_f2 G_f2 B_f2")

# The 4x6 matrix in the matting equation
A = Matrix([[1, 0, 0, -R_k1],
            [0, 1, 0, -G_k1],
            [0, 0, 1, -B_k1],
            [1, 0, 0, -R_k2],
            [0, 1, 0, -G_k2],
            [0, 0, 1, -B_k2]])

# The 1x6 matrix in the matting equation
b = Matrix([[R_f1 - R_k1],
            [G_f1 - G_k1],
            [B_f1 - B_k1],
            [R_f2 - R_k2],
            [G_f2 - G_k2],
            [B_f2 - B_k2]])

# Get the transpose of A
A_T = A.transpose()

# Calculate the pseudo-inverse of A = (A^T * A)^(-1) * A^T
A_pinv = ((A_T * A) ** -1)* A_T

# Calculate x ~= (A^+) * b
# Let x = [Rn, Gb, Bn, alpha]
x = A_pinv * b

# Factor x to simplify
fact_x = factor(x)

R_kd = R_k1 - R_k2
G_kd = G_k1 - G_k2
B_kd = B_k1 - B_k2
R_fd = R_f1 - R_f2
G_fd = G_f1 - G_f2
B_fd = B_f1 - B_f2

# We note that the following was factored out of each element of x:
# ((R_k1 ** 2 + 2 * (R_k1 * R_k2) + R_k2 ** 2) +
#  (G_k1 ** 2 + 2 * (G_k1 * G_k2) + G_k2 ** 2) +
#  (B_k1 ** 2 + 2 * (B_k1 * B_k2) + B_k2 ** 2)) ** -1
# Which is equivalent to:
common_x_fact = (R_kd ** 2 + \
                 G_kd ** 2 + \
                 B_kd ** 2) ** -1

Rn_top = (R_k1 + R_k2)*((B_fd * B_kd) + (G_fd * G_kd)) - \
         (R_f1 + R_f2) * (B_kd ** 2 + G_kd ** 2) + \
         2 * R_kd * (R_f1 * R_k2 - R_f2 * R_k1)

Gn_top = (G_k1 + G_k2)*((R_fd * R_kd) + (B_fd * B_kd)) - \
         (G_f1 + G_f2) * (R_kd ** 2 + B_kd ** 2) + \
         2 * G_kd * (G_f1 * G_k2 - G_f2 * G_k1)

Bn_top = (B_k1 + B_k2) * ((R_fd * R_kd) + (G_fd * G_kd)) - \
         (B_f1 + B_f2) * (R_kd ** 2 + G_kd ** 2) + \
         2 * B_kd * (B_f1 * B_k2 - B_f2 * B_k1)



output_latex(simplify(fact_x[0] + (Rn_top * common_x_fact) / 2))
output_latex(simplify(fact_x[1] + (Gn_top * common_x_fact) / 2))
output_latex(simplify(fact_x[2] + (Bn_top * common_x_fact) / 2))


# Define inner_x such that x = common_x_fact * inner_x -> inner_x = x / common_x_fact
inner_x = x / common_x_fact
inner_x = factor(inner_x)

# Note that we can factor (-1/2) out of inner_x ->
#   inner_x = (-1/2) * (inner_x2) ->
#   inner_x2 = inner_x / (-1/2) ->
#   inner_x2 = -2 * inner_x
inner_x2 = -2 * inner_x

# Solving for alpha
alpha_inner2 = inner_x2[-1]

# alpha_inner2 is currently too large to effectively describe here, so we have simplified it to:
# alpha_inner2 = a1 + a2 + a3 + a4 + a5 + a6, where:
a1 = 2 * (B_f1 - B_f2) * (B_k1 - B_k2)
a2 = 2 * (R_f1 - R_f2) * (R_k1 - R_k2)
a3 = 2 * (G_f1 - G_f2) * (G_k1 - G_k2)
a4 = -2 * (B_k1 - B_k2) ** 2
a5 = -2 * (R_k1 - R_k2) ** 2
a6 = -2 * (G_k1 - G_k2) ** 2

# Note that we can write alpha_inner2 as being composed of two parts:
alpha_inner_half1 = (a1 + a2 + a3)
alpha_inner_half2 = (a4 + a5 + a6)

# Note that for the final alpha:
# alpha = alpha_inner2 * (-1/2) * common_x_fact
# alpha = (alpha_inner_half1 + alpha_inner_half2) * (-1/2) * common_x_fact
# alpha = ((alpha_inner_half1) * (-1/2) * common_x_fact) + ((alpha_inner_half2) * (-1/2) * common_x_fact)

# Let's assign variables to each of these halves:
alpha_half1 = simplify(alpha_inner_half1 * (-1/2) * common_x_fact)
alpha_half2 = simplify(alpha_inner_half2 * (-1/2) * common_x_fact)

# More simplifying
alpha_half1 = nsimplify(alpha_half1)
alpha_half2 = nsimplify(alpha_half2)

# If we take each half and let:
# R_kd = R_k1 - R_k2
# G_kd = G_k1 - G_k2
# B_kd = B_k1 - B_k2
# R_fd = R_f1 - R_f2
# G_fd = G_f1 - G_f2
# B_fd = B_f1 - B_f2
# nume = ((R_fd * R_kd) + (G_fd * G_kd) + (B_fd * B_kd))
# denom = (R_kd ** 2) + (G_kd ** 2) + (B_kd ** 2)

# Then we can re-phrase each-half as:
# alpha_half1 = -nume / denom
# alpha_half2 = 1

# Since we verified above that alpha = alpha_half1 + alpha_half2:
alpha = alpha_half2 + alpha_half1 # addition is commutative
alpha = nsimplify(alpha)

# Which means: alpha = 1 - (nume / denom)

# WE HAVE SHOWN THE FOLLOWING:
# If:
#   complement = ((R_fd * R_kd) + (G_fd * G_kd) + (B_fd * B_kd)) / (R_kd ** 2) + (G_kd ** 2) + (B_kd ** 2)
# Then:
#   alpha = 1 - complement

# We will now expand the matting equation to show that we only need alpha to calculate what we need
(R_n, G_n, B_n, alpha) = symbols("R_n G_n B_n alpha")

composite_vector = Matrix([[R_n],
                           [G_n],
                           [B_n],
                           [alpha]])

matting_vector = A * composite_vector
# Note that matting_vector is equivalent to vector b, so:
R_n1 = solve(matting_vector[0] - b[0], R_n)
G_n1 = solve(matting_vector[1] - b[1], G_n)
B_n1 = solve(matting_vector[2] - b[2], B_n)
R_n2 = solve(matting_vector[3] - b[3], R_n)
G_n2 = solve(matting_vector[4] - b[4], G_n)
B_n2 = solve(matting_vector[5] - b[5], B_n)
answers = Matrix([[R_n1],
                  [G_n1],
                  [B_n1],
                  [R_n2],
                  [G_n2],
                  [B_n2]])

# We have obtained the answers:
# For color channel C and (f, k) fixed as (f1, k1) or (f2, k2), in general:
# C_n = C_f + C_k * alpha - C_k
#   -> C_n = C_f + C_k * (alpha - 1)
answers = Matrix([[R_f1 + R_k1 * (alpha - 1)],
                  [G_f1 + G_k1 * (alpha - 1)],
                  [B_f1 + B_k1 * (alpha - 1)],
                  [R_f2 + R_k2 * (alpha - 1)],
                  [G_f2 + G_k2 * (alpha - 1)],
                  [B_f2 + B_k2 * (alpha - 1)]])