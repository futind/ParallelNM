#include <iostream>
#include <cstdint>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <memory>

void GaussianElimination(double* L, double* U, int N, int pivot_idx, int border);
void standard_LU_Decomposition(double* A, double* L, double* U, int N, int distance_from_start, int current_block_size);
void calculateLUsubblocks(double* A, double* L, double* U, int N, int distance_from_start, int current_block_size);
void calculateA22(double* A, double* L, double* U, int N, int distance_from_start, int current_block_size, int multiplication_block_size);
void LU_Decomposition(double* A, double* L, double* U, int n);

// function that performs a step of Gaussian elimination
void GaussianElimination(double* L, double* U, int N, int pivot_idx, int border)
{
	// pivot_idx is the index of the column in which we need to set to zero
	// elements in the rows under the pivot_idx element

	// setting the corresponding element on a diagonal to one
	L[pivot_idx * N + pivot_idx] = 1;

	#pragma omp parallel for
	// i - is the index of a row
	for (int i = pivot_idx + 1; i < border; ++i)
	{
		// calculating the Gaussian multiplier for the i-th row
		double mu = U[i * N + pivot_idx] / U[pivot_idx * N + pivot_idx];

		// setting the elements in the rows under the pivot_idx-th in the pivot_idx-th row 
		// to mu in L, and to 0 in U
		L[i * N + pivot_idx] = mu;
		U[i * N + pivot_idx] = 0;

		// now, subtracting the pivot_idx-th row from the i-th row
		for (int j = pivot_idx + 1; j < border; ++j)
		{
			U[i * N + j] -= U[pivot_idx * N + j] * mu;
		}
	}
}

// function perfoming the standard LU decompoistion using Gaussian elimination
void standard_LU_Decomposition(double* A, double* L, double* U, int N, int distance_from_start, int current_block_size)
{
	#pragma omp parallel for collapse(2) schedule(guided)
	// filling the U and L matrices with the values from the matrix A and zeroes respectively
	for (int i = distance_from_start; i < distance_from_start + current_block_size; ++i)
	{
		for (int j = distance_from_start; j < distance_from_start + current_block_size; ++j)
		{
			U[i * N + j] = A[i * N + j];
			L[i * N + j] = 0.0;
		}
	}

	// Performing Gaussian elimination
	for (int i = distance_from_start; i < distance_from_start + current_block_size; ++i)
	{
		if (U[i * N + i] == 0)
		{
			throw std::logic_error("DIVIDING ON ZERO!");
		}
		GaussianElimination(L, U, N, i, distance_from_start + current_block_size);
	}
}

// functuion that calculates the submatrices L_21 and U_12
// by solving triange systems with multiple vectors of constans
// using the substitution method (forward for L_21 and back for U_12)
// by the time of use of this function L_11 and U_11 must have been calculated!
void calculateLUsubblocks(double* A, double* L, double* U, int N, int distance_from_start, int current_block_size)
{
	// i is the index of a row in U_12 and the index of a column in L_21
	for (int i = distance_from_start; i < distance_from_start + current_block_size; ++i)
	{
		#pragma omp parallel for
		// k is the index of a row in L_21 an of a column in U_12
		for (int k = distance_from_start + current_block_size; k < N; ++k)
		{
			double u = A[i * N + k];
			double l = A[k * N + i];

			// j is the index of a row in U_12 and the index of a column in L_21
			// basically previously calculated elements of k-th row/column in L_21/U_12
			for (int j = distance_from_start; j < i; ++j)
			{
				// subtracting previous elements of a k-th column
				u -= L[i * N + j] * U[j * N + k];
				// subtracting previous elements of a k-th row
				l -= L[k * N + j] * U[j * N + i];
			}
			U[i * N + k] = u;
			L[k * N + i] = l / U[i * N + i];
		}
	}
}

// function that calculates the reduced sumatrix matrix A_22 in place (in the matrix A itself)
// it basically multiplicates matrices L_21 and U_12 and subtracts the result 
// from the originalsubmatrix A_22 to form the A matrix suitable 
// for the next step of block LU decomposution
// by the time of use of this function L_21 and U_12 must have been calculated!
void calculateA22(double* A, double* L, double* U, int N,
				  int distance_from_start,
				  int current_block_size,
				  int multiplication_block_size)
{	
	// we are using the block algorithm of the matrix multiplication, meaning that we 
	// split the initial matrix (in our case submatrices L_21 and U_12) into square submatrices
	// with the dimensions (multiplication_block_size x multiplication_block_size) 
	// (mostly, because if block_i (or block_j) < N - distance_from_start - current_block_size 
	// is less than multiplication_block_size, the multiplication_block_size will be limited by that value)
	// and multiply corresponding blocks

	#pragma omp parallel for collapse(3) schedule(guided)
	// block_i is the index of a row in the block matrix L_21 and block matrix A_22
	for (int16_t block_i = distance_from_start + current_block_size; block_i < N; block_i += multiplication_block_size)
	{
		// block_j is the index of a column in the block matrix U_12 and block matrix A_22
		for (int16_t block_j = distance_from_start + current_block_size; block_j < N; block_j += multiplication_block_size)
		{
			// block_k is the index of a row in the block matrix U_12 and the index of a column in the block matrix L_21
			for (int16_t block_k = distance_from_start; block_k < distance_from_start + current_block_size; block_k += multiplication_block_size)
			{
				// now we use the algoritgm of matrix multiplication by the very definition
				// to multiplicate L_21_block_i,block_k on U_12_block_k,block_j

				// i - is the index of a row in our block matrces L_21 and A_22
				for (int16_t i = block_i; i < std::min(block_i + multiplication_block_size, N); ++i)
				{
					// j - is the index of a column in our block matrices U_12 and A_22
					for (int16_t j = block_j; j < std::min(block_j + multiplication_block_size, N); ++j) 
					{
						double tempAij = A[i * N + j];
						// k - is the index of a row in our block matrix U_12 and the index of a column in L_21
						for (int16_t k = block_k; k < std::min(block_k + multiplication_block_size, distance_from_start + current_block_size); ++k)
						{
							tempAij -= L[i * N + k] * U[k * N + j];
						}
						A[i * N + j] = tempAij;
					}
				}
			}
		}
	}
}

void LU_Decomposition(double* A, double* L, double* U, int n)
{
	if (n < 2) return;
	// current_block_size is current block size of block LU decomposition
	int16_t current_block_size = 1024;
	// multiplication_block_size describes the dimensions of square block, which is used in block algorithm
	// of a matrix multiplication
	int16_t multiplication_block_size = 256;
	// distance_from_start is an offset from the start of the row and the start of the column
	int16_t distance_from_start = 0;

	if (current_block_size > n) current_block_size = n / 2;
	if (current_block_size < multiplication_block_size) multiplication_block_size = current_block_size;
	

	for (int times = 0; times < n / current_block_size; ++times)
	{
		standard_LU_Decomposition(A, L, U, n, distance_from_start, current_block_size);
		calculateLUsubblocks(A, L, U, n, distance_from_start, current_block_size);
		calculateA22(A, L, U, n, distance_from_start, current_block_size, multiplication_block_size);
		distance_from_start += current_block_size;
	}

	if (distance_from_start < n)
	{
		current_block_size = n - distance_from_start;
		standard_LU_Decomposition(A, L, U, n, distance_from_start, current_block_size);
	}
}

void matrixMultiplication(double* A, int rowsA, int columnsA,
						  double* B, int rowsB, int columnsB,
						  double* C, int rowsC, int columnsC,
						  int multiplication_block_size)
{
	if (std::min({ rowsA, columnsA, rowsB, columnsB }) <= 0)
	{
		throw std::logic_error("MATRICES DIMENSIONS ARE NEGATIVE!");
	}

	if (columnsA != rowsB)
	{
		throw std::logic_error("MATRICES DIMENSIONS ARE NOT CONFORMABLE FOR MULTIPLICATION!");
	}

	if (rowsA != rowsC || columnsB != columnsC)
	{
		throw std::logic_error("THE RESULTING MATRIX HAS INCORRECT DIMENSIONS!");
	}

	if (multiplication_block_size > std::min({ rowsA, columnsA, rowsB, columnsB }))
	{
		multiplication_block_size = std::min({ rowsA, columnsA, rowsB, columnsB });
	}

	#pragma omp parallel for collapse(3) schedule(guided)
	for (int bi = 0; bi < rowsA; bi += multiplication_block_size)
	{
		for (int bj = 0; bj < columnsB; bj += multiplication_block_size)
		{
			for (int bk = 0; bk < columnsA; bk += multiplication_block_size)
			{
				for (int i = bi; i < std::min(bi + multiplication_block_size, rowsA); ++i)
				{
					for (int j = bj; j < std::min(bj + multiplication_block_size, columnsB); ++j)
					{
						for (int k = bk; k < std::min(bk + multiplication_block_size, columnsA); ++k)
						{
							C[i * columnsC + j] += A[i * columnsA + k] * B[k * columnsB + j];
						}
					}
				}
			}
		}
	}
}

void matrixDifference(double* A, int rowsA, int columnsA,
					  double* B, int rowsB, int columnsB,
					  double* C, int rowsC, int columnsC)
{
	if (!(rowsA == rowsB && columnsA == columnsB))
	{
		throw std::logic_error("MATRICES DIMENSIONS ARE NOT COMFORMABLE FOR ADDITION!");
	}
	if (!(rowsA == rowsC && columnsA == columnsC))
	{
		throw std::logic_error("THE RESULTING MATRIX HAS INCORRENT DIMENSIONS");
	}
	#pragma omp parallel for
	for (int i = 0; i < rowsA; ++i)
	{
		for (int j = 0; j < columnsA; ++j)
		{
			C[i * columnsA + j] = A[i * columnsA + j] - B[i * columnsA + j];
		}
	}

}

double eucledian_norm(double* A, int n)
{
	double sum = 0;
	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < n; ++j)
		{
			sum += std::pow(std::abs(A[i * n + j]), 2);
		}
	}
	return std::sqrt(sum);
}

void zeroMatrix(double* M, int N)
{
	#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			M[i * N + j] = 0.0;
		}
	}
}

void showMatrix(double* M, int N, std::string name)
{
	std::cout << "Matrix " << name << std::endl;
	std::cout.precision(16);
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{	
			std::cout << std::setw(18) << M[i * N + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

void RNGMatrix(double* M, int N)
{
	std::uniform_real_distribution<double> uniform(2, 10000);
	std::default_random_engine dre(std::chrono::system_clock::now().time_since_epoch().count());
	#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			M[i * N + j] = uniform(dre);
		}
	}
}

void CopyMatrix(double* FROM, double* INTO, int N)
{
	#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < N; ++j)
		{
			INTO[i * N + j] = FROM[i * N + j];
		}
	}
}

int main()
{
	const int N = 3120;

	// real matrix A
	double* A = new double[N * N];

	// matrices A, L and U containing the result of the 
	// block version of the LU_decomposition algorithm
	double* A_p = new double[N * N];
	double* L_p = new double[N * N];
	double* U_p = new double[N * N];
	// initializing L_p and U_p with zeroes
	zeroMatrix(L_p, N);
	zeroMatrix(U_p, N);

	// matrices A, L and U containing the result of the 
	// standard version of the LU_decomposition algorithm
	double* A_s = new double[N * N];
	double* L_s = new double[N * N];
	double* U_s = new double[N * N];
	// initializing L_S and U_S with zeroes
	zeroMatrix(L_s, N);
	zeroMatrix(U_s, N);

	// Initializing the original A matrix with random generated values
	RNGMatrix(A, N);
	// Copying the original matrix A into block and standard versions
	CopyMatrix(A, A_p, N);
	CopyMatrix(A, A_s, N);

	// Showing the original matrix A
	//showMatrix(A, N, "A");

	// Performing standard LU decomposition
	auto ts1 = std::chrono::high_resolution_clock::now();
	standard_LU_Decomposition(A_s, L_s, U_s, N, 0, N);
	auto ts2 = std::chrono::high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	std::chrono::duration<double, std::milli> ms_double = ts2 - ts1;

	std::cout << "PARALLEL STANDARD: " << ms_double.count() << "ms\n";

	// Performing block LU decomposition
	auto t1 = std::chrono::high_resolution_clock::now();
	LU_Decomposition(A_p, L_p, U_p, N);
	auto t2 = std::chrono::high_resolution_clock::now();
	/* Getting number of milliseconds as a double. */
	std::chrono::duration<double, std::milli> mp_double = t2 - t1;

	std::cout << "PARALLEL BlOCK: " << mp_double.count() << "ms\n";

	// Now, lets show L_s, L_p
	//showMatrix(L_s, N, "L_s");
	//showMatrix(L_p, N, "L_p");

	// Now, lets show U_s, U_p
	//showMatrix(U_s, N, "U_s");
	//showMatrix(U_p, N, "U_p");

	// Now lets multiplicate L_s and U_s
	// We will save the result of that multiplication into matrix A_s
	zeroMatrix(A_s, N);
	matrixMultiplication(L_s, N, N, U_s, N, N, A_s, N, N, 64);
	// Now, lets subract matrix A_s from matrix A, and take its norm
	double* A_s_diff = new double[N * N];
	matrixDifference(A_s, N, N, A, N, N, A_s_diff, N, N);
	std::cout << "||A - L_s * U_s|| = " << std::setprecision(8) << eucledian_norm(A_s_diff, N) << std::endl;

	// Now lets do the same for L_p and U_p
	zeroMatrix(A_p, N);
	matrixMultiplication(L_p, N, N, U_p, N, N, A_p, N, N, 64);
	double* A_p_diff = new double[N * N];
	matrixDifference(A_p, N, N, A, N, N, A_p_diff, N, N);
	std::cout << "||A - L_p * U_p|| = " << std::setprecision(8) << eucledian_norm(A_p_diff, N) << std::endl;

	// calculating and showing euclidian norm of the original matrix, 
	std::cout << "||A|| = " << std::setprecision(16) << eucledian_norm(A, N) << std::endl;
	std::cout << "||A_s|| = " << std::setprecision(16) << eucledian_norm(A_s, N) << std::endl;
	std::cout << "||A_p|| = " << std::setprecision(16) << eucledian_norm(A_p, N) << std::endl;

	// now, calculating and showing the result of ||A - L_s * U_s|| / ||A||
	std::cout << "||A - L_s * U_s|| / ||A|| = " << std::setprecision(16) << eucledian_norm(A_s_diff, N) / eucledian_norm(A, N) << std::endl;
	// now, calculating and showing the result of ||A - L_p * U_p|| / ||A||
	std::cout << "||A - L_p * U_p|| / ||A|| = " << std::setprecision(16) << eucledian_norm(A_p_diff, N) / eucledian_norm(A, N) << std::endl;

	delete[] A;
	delete[] A_s;
	delete[] A_p;
	delete[] L_s;
	delete[] L_p;
	delete[] A_s_diff;
	delete[] A_p_diff;

	return 0;
}