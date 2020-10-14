void levelSelect(int &level, const int cycle, const double residual_before, const double residual_after, const int numSchemes)
{
	if (cycle == 0) {
		level = 0;
	}
	else{
		if ((residual_after / residual_before > 0.4) && (level < numSchemes-1)) {
			level = level + 1;
		}
		else if ((residual_after / residual_before < 0.4) && (residual_after / residual_before > 0.2) && (level > 0)) {	
			level = level - 1;
		}
		else {
			level = level;
		}
	}
}
