def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	if len(a) != len(b):
		return -1
	else:
		out = []
		for row in a:
			s = 0.0
			for elem1, elem2 in zip(row, b):
				s += elem1 * elem2
			out.append(s)
		return out