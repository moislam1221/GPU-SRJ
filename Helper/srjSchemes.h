void loadSRJSchemes(double * srjSchemes, const int numSchemeParams)  
{
	// Load the SRJ Schemes txt files from Python
	std::ifstream srjSchemeParametersFile("srjSchemeParameters.txt");
	for (int i = 0; i < numSchemeParams; i++) {
		srjSchemeParametersFile >> srjSchemes[i];
	}
}

void loadIndexPointer(int * indexPointer, const int numSchemes)
{
	// Load the size of schemes txt files from Python
	double * sizeOfSchemeDouble = new double[numSchemes];
	int * sizeOfScheme = new int[numSchemes];
	std::ifstream sizeOfSchemeFile("sizeOfScheme.txt");
	for (int i = 0; i < numSchemes; i++) {
		sizeOfSchemeFile >> sizeOfSchemeDouble[i];
		sizeOfScheme[i] = int(sizeOfSchemeDouble[i]);
	}

	// Create the index pointer array based on scheme sizes
	for (int i = 1; i < numSchemes+1; i++) {
		for (int k = 0; k < i; k++) {
			indexPointer[i] = indexPointer[i] + sizeOfScheme[k];
		}
	}
}
