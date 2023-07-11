/*
 * Macro to batch scale a dir of images by 2 in Z
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
	// get the file name without the extension
	dotIndex = indexOf(file, "."); 
	if (dotIndex > 0) {
		fileNameWithoutExtension = substring(file, 0, dotIndex);
	}else {
		print("Error finding the file name without the extension");
	}
	run("Scale...", "x=1.0 y=1.0 z=1.5 interpolation=Bicubic process");
	newname = fileNameWithoutExtension + "_scaleZ.tif";
	saveAs("Tiff", output + "/" + newname);
	close(newname);
	close(file);

	
	
	
	
}






