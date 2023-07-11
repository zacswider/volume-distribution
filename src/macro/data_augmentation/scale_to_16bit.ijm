/*
 * Macro uses the max projection of a 32-bit de-noised stack to estimate
 * the appropriate dyanmic range to scale back to 16-bit
 */
 
#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix

processFolder(input)

// function to scan folders/subfolders/files to find files with correct suffix
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		print(list[i]);
		if(File.isDirectory(input + File.separator + list[i])){
			print(list[i] + " is a directory");
		}
		if(list[i].contains("Max")){
			print(list[i] + " is a max projection");
		}
		if(endsWith(list[i], suffix)){
			processFile(input, output, list[i]);
		}
	}
}

function processFile(input, output, file) {
	// open stack and split the channels
	print("Opening " + input + "/" + file);
	open(input + "/" + file);
	run("Z Project...", "projection=[Max Intensity]");
	getStatistics(maxarea, maxmean, maxmin, maxmax, maxstd, maxhistogram);
	close("MAX_" + file);
	selectWindow(file);
	run("Z Project...", "projection=[Min Intensity]");
	getStatistics(minarea, minmean, minmin, minmax, minstd, minhistogram);
	close("MIN_" + file);
	selectWindow(file);
	setMinAndMax(minmin, maxmax);
	run("16-bit");
	// get the file name without the extension
	dotIndex = indexOf(file, "."); 
	if (dotIndex > 0) {
		fileNameWithoutExtension = substring(file, 0, dotIndex);
	} else {
		print("no dot index to parse...") ;
	}	
	saveAs("Tiff", output + "/" + fileNameWithoutExtension + "_16bit.tif");
	close(fileNameWithoutExtension + "_16bit.tif");
}






