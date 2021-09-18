import React, {useState, useMemo, useCallback, useEffect} from 'react';
import {useDropzone} from 'react-dropzone';
import Button from '@mui/material/Button'; 
import axios from 'axios';

const baseStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out'
};

const activeStyle = {
  borderColor: '#2196f3'
};

const acceptStyle = {
  borderColor: '#00e676'
};

const rejectStyle = {
  borderColor: '#ff1744'
};

const thumbsContainer = {
    display: 'flex',
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 16
  };
  
  const thumb = {
    display: 'inline-flex',
    borderRadius: 2,
    border: '1px solid #eaeaea',
    marginBottom: 8,
    marginRight: 8,
    width: 100,
    height: 100,
    padding: 4,
    boxSizing: 'border-box'
  };
  
  const thumbInner = {
    display: 'flex',
    minWidth: 0,
    overflow: 'hidden'
  };
  
  const img = {
    display: 'block',
    width: 'auto',
    height: '100%'
  };

const Basic = (props) => {
    const [imageUrl, setImageUrl] = useState('');
    const [files, setFiles] = useState([]);

    // Handlear el archivo subido 
    const onDrop = useCallback((acceptedFiles) => {
        setFiles(acceptedFiles.map(file => Object.assign(file, {
            preview: URL.createObjectURL(file)
        })));
        acceptedFiles.forEach((file) => {
            console.log("FILE:" + JSON.stringify(file));
            setImageUrl(file.path);
            const reader = new FileReader();

            // reader.onabort = () => console.log('file reading was aborted')
            // reader.onerror = () => console.log('file reading has failed')
            reader.onload = () => {
            // Do whatever you want with the file contents
                const dataURL = reader.result;
                const output = document.getElementById('image_loaded');
                output.src = dataURL;
                setFiles(arr => [...arr, dataURL])
            }
            reader.readAsArrayBuffer(file)
            // reader.readAsDataURL(file);
        })}, [])

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject
  } = useDropzone({accept: 'image/*', onDrop});

  const style = useMemo(() => ({
    ...baseStyle,
    ...(isDragActive ? activeStyle : {}),
    ...(isDragAccept ? acceptStyle : {}),
    ...(isDragReject ? rejectStyle : {})
  }), [
    isDragActive,
    isDragReject,
    isDragAccept
  ]);

  const thumbs = files.map(file => (
    <div style={thumb} key={file.name}>
      <div style={thumbInner}>
        <img
          src={file.preview}
          style={img}
        />
      </div>
    </div>
  ));

  useEffect(() => () => {
    // Make sure to revoke the data uris to avoid memory leaks
    files.forEach(file => URL.revokeObjectURL(file.preview));
  }, [files]);

  return (
    <div className="container">
      <div {...getRootProps({style})}>
        <input {...getInputProps()} />
        <p>Arrastra y suelta la retinografía en esta región.</p>
      </div>
      <aside style={thumbsContainer}>
        {thumbs}
      </aside>
      <Button onClick={() => {
          console.log("FILE 0: " + JSON.stringify(files[0]))
          axios({
            method: 'post',
            url: 'endpoint-backend',
            responseType: 'stream'
          })
            .then(function (response) {
            //   response.data.pipe(fs.createWriteStream('44349_left.jpeg'))
            });   
        }} variant="outlined">Enviar</Button>
        <img width="800px" id="image_loaded"/>
    </div>
  );
}

export default Basic;