
/**
 * @brief Integrate with trapezoid rule on CPU. 
 * 
 * @param n     (int) number of sections to divide the funtion into.
 * @param f     (array) function samples
 * @param dx    (array) an array of length n containing the width of each section. 
*/
float integrate_cpu(int n, float f[], float dx[]);

float init_vector(float a, float b, int num_sections, std::vector<float> &v);
