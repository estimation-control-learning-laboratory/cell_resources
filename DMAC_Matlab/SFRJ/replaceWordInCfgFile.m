function replaceWordInCfgFile(inputFilePath, outputFilePath, targetWord, replacementWord)
    % Reads the content of the config file and changes any target word to
    % your desired word. 
    try
        fileContent = fileread(inputFilePath);
    catch
        error('Error reading the input file.');
    end
    
    modifiedContent = strrep(fileContent, targetWord, replacementWord);
    
    try
        fid = fopen(outputFilePath, 'w');
        fwrite(fid, modifiedContent);
        fclose(fid);
    catch
        error('Error writing to the output file.');
    end
    
    disp('Word replacement successful.');
end