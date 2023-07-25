#include <bits/stdc++.h>
using namespace std;

int main(int argc, char const *argv[]) {
  ifstream test_in(argv[1]);  /* This stream reads from test's input file   */
  ifstream test_out(argv[2]); /* This stream reads from test's output file  */
  ifstream user_out(argv[3]); /* This stream reads from user's output file  */

  int n;
  test_in >> n;
  int u[n][n];
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      test_in >> u[i][j];

  double s1[n], s2[n];
  double s = 0;
  for (int i = 0; i < n; i++) {
    user_out >> s1[i];
    s += s1[i];
  }
  if ((s - 1) * (s - 1) > 1e-6) {
    return 1; // Invalid probabilities.
  }

  s = 0;
  for (int i = 0; i < n; i++) {
    user_out >> s2[i];
    s += s2[i];
  }
  if ((s - 1) * (s - 1) > 1e-6) {
    return 1; // Invalid probabilities.
  }

  double mn = 101, mx = 0;
  for (int i = 0; i < n; i++) {
    double s = 0;
    for (int j = 0; j < n; j++) {
      s += s1[j] * u[j][i];
    }
    if (s < mn) {
      mn = s; // Minimum reward for p1.
    }
  }

  for (int i = 0; i < n; i++) {
    double s = 0;
    for (int j = 0; j < n; j++) {
      s += s2[j] * u[i][j];
    }
    if (s > mx) {
      mx = s; // Maximum reward for p1.
    }
  }

  if ((mx - mn) * (mx - mn) > 1e-6) {
    return 1; // Strategies don't form a nash equilibrium
  }
  return 0;
}