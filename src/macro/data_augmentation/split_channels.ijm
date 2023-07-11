/*
 * Macro to accept a dir of images, open the hyperstacks and split into a "_PI"
 * and "_Tub" channel and save them into a subdir.
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
	run("Split Channels");
	for (i = 1; i <= 2; i++) {
		selectWindow("C" + i + "-" + file);
		// get the file name without the extension
		dotIndex = indexOf(file, "."); 
		if (dotIndex > 0) {
			fileNameWithoutExtension = substring(file, 0, dotIndex);
		} else {
			print("no dot index to parse...") ;
		}	
		if(i==1) {
			saveAs("Tiff", output + "/" + fileNameWithoutExtension + "_PI.tif");
		}
		if(i==2) {
			saveAs("Tiff", output + "/" + fileNameWithoutExtension + "_Tub.tif");
		}
	}
	close(fileNameWithoutExtension + "_PI.tif");
	close(fileNameWithoutExtension + "_Tub.tif");

	
	
	
	
}






