#include <iostream>
#include <cstdint>
#include <cmath>
#include <random>
#include <iomanip>
#include <chrono>
#include <memory>
#include <algorithm>
#include <omp.h>

void sLUinplace_standalone(double* A, double* L, double* U, int N);

void sLUinplace(double* A, int N, int dA, int r);
void LUsubinplace(double* A, int N, int dA, int r);
void A22inplace(double* A, int N, int dA, int r, int mbc);
void LU_Decomposition(double* A, double* L, double* U, int n, int bs, int mbs);

void matrixMultiplication(double* A, int rowsA, int columnsA,
	double* B, int rowsB, int columnsB,
	double* C, int rowsC, int columnsC,
	int multiplication_block_size);
void matrixDifference(double* A, int rowsA, int columnsA,
	double* B, int rowsB, int columnsB,
	double* C, int rowsC, int columnsC);
double eucledian_norm(double* M, int rows, int columns);
void zeroMatrix(double* M, int rows, int columns);
void showMatrix(double* M, int rows, int columns, std::string name);
void RNGMatrix(double* M, int rows, int columns);
void CopyMatrix(double* FROM, double* INTO, int rows, int columns);

void sLUinplace_standalone(double* A, double* L, double* U, int N)
{

	for (int i = 0; i < N; ++i)
	{
		#pragma omp parallel for 
		for (int j = i + 1; j < N; ++j)
		{
			double mu = A[j * N + i] / A[i * N + i];

			A[j * N + i] = mu;

			for (int k = i + 1; k < N; ++k)
			{
				A[j * N + k] -= A[i * N + k] * mu;
			}
		}
	}

	#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		for (int j = i; j < N; ++j)
		{
			U[i * N + j] = A[i * N + j];
			L[i * N + j] = 0.0;
		}
	}

	#pragma omp parallel for
	for(int i = 0; i < N; ++i)
	{
		L[i * N + i] = 1;
	}

	#pragma omp parallel for
	for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			L[i * N + j] = A[i * N + j];
			U[i * N + j] = 0.0;
		}
	}
}

void sLUinplace(double* A, int N, int dA, int r)
{

	for (int i = dA; i < dA + r; ++i)
	{
		#pragma omp parallel for 
		for (int j = i + 1; j < dA + r; ++j)
		{
			double mu = A[j * N + i] / A[i * N + i];

			A[j * N + i] = mu;

			for (int k = i + 1; k < dA + r; ++k)
			{
				A[j * N + k] -= A[i * N + k] * mu;
			}
		}
	}
}

void LUsubinplace(double* A, int N, int dA, int r)
{

	for (int i = dA; i < dA + r; ++i)
	{
		#pragma omp parallel for
		for (int k = dA + r; k < N; ++k)
		{
			double u = A[i * N + k];
			double l = A[k * N + i];
			for (int j = dA; j < i; ++j)
			{
				// subtracting previous elements of a k-th column
				u -= A[i * N + j] * A[j * N + k];
				// subtracting previous elements of a k-th row
				l -= A[k * N + j] * A[j * N + i];
			}
			A[i * N + k] = u;
			A[k * N + i] = l / A[i * N + i];
		}
	}
}


void A22inplace(double* A, int N, int dA, int r, int mbc)
{
	#pragma omp parallel for
	for (int bi = dA + r; bi < N; bi += mbc)
	{
		for (int bj = dA + r; bj < N; bj += mbc)
		{
			for (int bk = dA; bk < dA + r; bk += mbc)
			{
				for (int i = bi; i < (bi + mbc > N ? N : bi + mbc); ++i)
				{
					for (int k = bk; k < (bk + mbc > dA + r ? dA + r : bk + mbc); ++k)
					{
						for (int j = bj; j < (bj + mbc > N ? N : bj + mbc); ++j)
						{
							A[i * N + j] -= A[i * N + k] * A[k * N + j];
						}
					}
				}
			}
		}
	}
}

void LU_Decomposition(double* A, double* L, double* U, int n, int bs, int mbs)
{
	// current_block_size is current block size of block LU decomposition
	int current_block_size = bs;
	// multiplication_block_size describes the dimensions of square block, which is used in block algorithm
	// of a matrix multiplication
	int multiplication_block_size = mbs;
	// distance_from_start is an offset from the start of the row and the start of the column
	int distance_from_start = 0;

	if (bs > n) current_block_size = n;
	if (mbs > current_block_size) multiplication_block_size = current_block_size;

	for (int times = 0; times < n / current_block_size; ++times)
	{
		sLUinplace(A, n, distance_from_start, current_block_size);
		LUsubinplace(A, n, distance_from_start, current_block_size);
		A22inplace(A, n, distance_from_start, current_block_size, multiplication_block_size);

		distance_from_start += current_block_size;
	}

	if (distance_from_start < n)
	{
		current_block_size = n - distance_from_start;
		sLUinplace(A, n, distance_from_start, current_block_size);
	}

	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		for (int j = i; j < n; ++j)
		{
			U[i * n + j] = A[i * n + j];
			L[i * n + j] = 0;
		}
		L[i * n + i] = 1;
	}

	#pragma omp parallel for
	for (int i = 0; i < n; ++i)
	{
		for (int j = 0; j < i; ++j)
		{
			L[i * n + j] = A[i * n + j];
			U[i * n + j] = 0;
		}
	}
}


// a function that calulates the prouct of matrices A and B
// which is saved into matrix C
void matrixMultiplication(double* A, int rowsA, int columnsA,
	double* B, int rowsB, int columnsB,
	double* C, int rowsC, int columnsC,
	int multiplication_block_size)
{

	if (columnsA != rowsB)
	{
		throw std::logic_error("MATRICES DIMENSIONS ARE NOT CONFORMABLE FOR MULTIPLICATION!");
	}

	if (rowsA != rowsC || columnsB != columnsC)
	{
		throw std::logic_error("THE RESULTING MATRIX HAS INCORRECT DIMENSIONS!");
	}

	zeroMatrix(C, rowsC, columnsC);

	#pragma omp parallel for
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

// a function that calculates the difference of matrices A and B
// which is saved into matrix C
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

// a function that calculates euclidian norm of the matrix M
double eucledian_norm(double* M, int rows, int columns)
{
	double sum = 0;
	#pragma omp parallel for
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			sum += M[i * columns + j] * M[i * columns + j];
		}
	}
	return std::sqrt(sum);
}

// a function that sets all the elements of the matrix M to zero
void zeroMatrix(double* M, int rows, int columns)
{
	#pragma omp parallel for
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			M[i * columns + j] = 0.0;
		}
	}
}

// a function that shows the matrix M using standard output
void showMatrix(double* M, int rows, int columns, std::string name)
{
	std::cout << "Matrix " << name << std::endl;
	std::cout.precision(16);
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			std::cout << std::setw(18) << M[i * columns + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// a function that assignes randomly generated values 
// to the elemets of the matrix M
void RNGMatrix(double* M, int rows, int columns)
{
	std::uniform_real_distribution<double> uniform(-10000, 10000);
	std::random_device rd;
	std::default_random_engine dre(rd());
	#pragma omp parallel for
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			M[i * columns + j] = uniform(dre);
		}
	}
}

// a function that copies one matrix from FROM to matrix INTO
void CopyMatrix(double* FROM, double* INTO, int rows, int columns)
{
#pragma omp parallel for
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < columns; ++j)
		{
			INTO[i * columns + j] = FROM[i * columns + j];
		}
	}
}

using Timer = std::chrono::time_point<std::chrono::high_resolution_clock>;

double getDuration(Timer start, Timer end)
{
	return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) * 1e-6;
}

int main()
{
	int number_of_pairs;
	std::cout << "INPUT OVERALL NUMBER OF TRIALS = ";
	std::cin >> number_of_pairs;

	std::vector<int> dimensions(number_of_pairs);
	std::vector<int> blocks(number_of_pairs);
	std::vector<int> multiplication_block(number_of_pairs);

	std::cout << "INPUT PAIRS (DIMENSION (N) and BLOCK_SIZE and MULTIPLICATION_BLOCK_SIZE): " << std::endl;

	for (int i = 0; i < number_of_pairs; ++i)
	{
		std::cin >> dimensions[i] >> blocks[i] >> multiplication_block[i];
	}

	std::vector<double> Trials_standard;
	std::vector<double> Trials_blocked;

	for (int i = 0; i < number_of_pairs; ++i)
	{
		int trials = 10;
		int N = dimensions[i];

		if (trials >= 2000) trials = 5;
		if (trials >= 3000) trials = 4;
		if (trials >= 4000) trials = 3;

		int BLOCK_SIZE = blocks[i];
		int MULT_BLOCK_SIZE = multiplication_block[i];

		double* A_s_redundancy = new double[N * N];
		double* A_s_diff = new double[N * N];
		double* A_s = new double[N * N];
		double* L_s = new double[N * N];
		double* U_s = new double[N * N];

		double* A_p_redundancy = new double[N * N];
		double* A_p_diff = new double[N * N];
		double* A_p = new double[N * N];
		double* L_p = new double[N * N];
		double* U_p = new double[N * N];

		
		for (int T = 1; T <= omp_get_max_threads(); ++T)
		{
			omp_set_num_threads(T);

			Trials_standard.clear();
			Trials_blocked.clear();

			int standard_err_counter = 0;
			int block_err_counter = 0;
			
			for (int i = 0; i < trials; ++i)
			{

				RNGMatrix(A_s, N, N);
				CopyMatrix(A_s, A_s_redundancy, N, N);
				RNGMatrix(A_p, N, N);
				CopyMatrix(A_p, A_p_redundancy, N, N);

				// STANDARD
				Timer standard_start = std::chrono::high_resolution_clock::now();
				sLUinplace_standalone(A_s, L_s, U_s, N);
				Timer standard_end = std::chrono::high_resolution_clock::now();

				matrixMultiplication(L_s, N, N, U_s, N, N, A_s, N, N, MULT_BLOCK_SIZE);
				// Now, lets subract matrix A_s from matrix A, and take its norm
				matrixDifference(A_s, N, N, A_s_redundancy, N, N, A_s_diff, N, N);
				if ((eucledian_norm(A_s_diff, N, N) / eucledian_norm(A_s_redundancy, N, N)) < 0.01)
				{
					Trials_standard.push_back(getDuration(standard_start, standard_end));
				}
				else
				{
					standard_err_counter++;
				}
				Trials_standard.push_back(getDuration(standard_start, standard_end));

				// BLOCKED
				Timer block_start = std::chrono::high_resolution_clock::now();
				LU_Decomposition(A_p, L_p, U_p, N, BLOCK_SIZE, MULT_BLOCK_SIZE);
				Timer block_end = std::chrono::high_resolution_clock::now();

				matrixMultiplication(L_p, N, N, U_p, N, N, A_p, N, N, MULT_BLOCK_SIZE);
				matrixDifference(A_p, N, N, A_p_redundancy, N, N, A_p_diff, N, N);
				if ((eucledian_norm(A_p_diff, N, N) / eucledian_norm(A_p_redundancy, N, N)) < 0.01)
				{
					Trials_blocked.push_back(getDuration(block_start, block_end));
				}
				else
				{
					standard_err_counter++;
				}
				Trials_blocked.push_back(getDuration(block_start, block_end));
			}
			std::sort(Trials_standard.begin(), Trials_standard.end());
			std::sort(Trials_blocked.begin(), Trials_blocked.end());

			double standard_min = Trials_standard[0];
			double standard_max = Trials_standard[Trials_standard.size() - 1];
			double standard_median = Trials_standard[Trials_standard.size() / 2];
			if (Trials_standard.size() % 2 == 0)
			{
				standard_median += Trials_standard[(Trials_standard.size() + 1) / 2];
				standard_median /= 2;
			}

			std::cout << "Standard LU Decomposition (N = " << N << "): " << std::endl;
			std::cout << "NUMBER OF THREADS = " << T << std::endl;
			std::cout << "Number of errors: " << standard_err_counter << std::endl;
			std::cout << "Min | Median | Max time : " << standard_min << " | "
				<< standard_median << " | "
				<< standard_max << std::endl;


			double blocked_min = Trials_blocked[0];
			double blocked_max = Trials_blocked[Trials_blocked.size() - 1];
			double blocked_median = Trials_blocked[Trials_blocked.size() / 2];
			if (Trials_blocked.size() % 2 == 0)
			{
				blocked_median += Trials_blocked[(Trials_blocked.size() + 1) / 2];
				blocked_median /= 2;
			}

			std::cout << "Blocked LU Decomposition (N = " << N
				<< ", BLOCK_SIZE = " << BLOCK_SIZE
				<< ", MULT_BLOCK_SIZE = " << MULT_BLOCK_SIZE << std::endl;
			std::cout << "NUMBER OF THREADS = " << T << std::endl;
			std::cout << "Number of errors: " << block_err_counter << std::endl;
			std::cout << "Min | Median | Max time : " << blocked_min << " | "
				<< blocked_median << " | "
				<< blocked_max << std::endl;

			std::cout << "----------------------------------------------" << std::endl;

		}

		std::cout << std::endl;
		std::cout << "**************************************************" << std::endl;
		std::cout << std::endl;

		delete[] A_s;
		delete[] A_p;
		delete[] A_p_diff;
		delete[] A_s_diff;
		delete[] A_s_redundancy;
		delete[] A_p_redundancy;
		delete[] L_s;
		delete[] L_p;
		delete[] U_s;
		delete[] U_p;
	}

	return 0;
}