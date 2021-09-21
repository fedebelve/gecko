import React, { useState } from 'react';

const ImageUpload64 = () => {
    const [selectedFile, setSelectedFile] = useState();
    const [isFilePicked, setIsFilePicked] = useState(false);

    const changeHandler = (event) => {
        setSelectedFile(event.target.files[0]);
        setIsFilePicked(true);
    };

    const handleSubmission = () => {
        const reader = new FileReader();
        reader.readAsDataURL(selectedFile);
        reader.onload = () => {
            console.log("FILE CONTENT: " + reader.result.slice(reader.result.indexOf(',') + 1, -1));
            fetch(`http://localhost:8000/analize/`, {
                method: 'POST',
                body: {
                    "worklist": [
                        {
                            "eye": "left",
                            "img_name": "reti11",
                            "img_bytes": reader.result.slice(reader.result.indexOf(',') + 1, -1)
                        }
                    ]
                },
                headers: {
                    "Authorization": "TOKEN f2e94d1c083f1f84117b5c515b028a2dcfc09776"
                }
            }).then(response => response.json())
            .then(result => console.log('Success:', result))
            .catch(error => console.log('Error:', error));
        }
    };

    return (
        <div>
            <input type="file" name="file" onChange={changeHandler}/>
            {isFilePicked ? (
                <div>
                    <p>Filename: {selectedFile.name}</p>
                    <p>Filetype: {selectedFile.type}</p>
                    <p>Size in bytes: {selectedFile.size}</p>
                </div>
            ) : (
                <p>Select a file to show details</p>
            )}
            <div>
                <button onClick={handleSubmission}>Submit</button>
            </div>
        </div>
    );
}

export default ImageUpload64;