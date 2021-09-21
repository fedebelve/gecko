import React, {useState, useMemo, useCallback, useEffect} from 'react';
import {useDropzone} from 'react-dropzone';
import Button from '@mui/material/Button'; 
import {baseStyle, activeStyle, acceptStyle, 
  rejectStyle, thumbsContainer, thumb, thumbInner, img} from './basic-styles';
import Alert from '@mui/material/Alert';
import AlertTitle from '@mui/material/AlertTitle';
import ImageViewer from './ImageViewer';

const Basic = (props) => {
    const [imageUrl, setImageUrl] = useState('');
    const [files, setFiles] = useState([]);
    const [file, setFile] = useState();
    const [result, setResult] = useState({});
    const [flagResult, setFlagResult] = useState(false);
    const [flagFileLoaded, setFlagFileLoaded] = useState(false);

    // Handlear el archivo subido 
    const onDrop = useCallback((acceptedFiles) => {
        // To generate files preview
        // setFiles(acceptedFiles.map(file => Object.assign(file, {
        //     preview: URL.createObjectURL(file)
        // })));
        // end
        setFlagFileLoaded(true);
        acceptedFiles.forEach((file) => {
            setFile(file);
            setImageUrl(file.path);
            const reader = new FileReader();

            reader.onabort = () => console.log('file reading was aborted')
            reader.onerror = () => console.log('file reading has failed')
            reader.onload = () => {
            // Do whatever you want with the file contents
                const dataURL = reader.result;
                const output = document.getElementById('image_loaded');
                output.src = dataURL;
                // setFiles(arr => [...arr, dataURL]);

                // Send file to backend
            }
            // reader.readAsBinaryString(file)
            reader.readAsDataURL(file);
            setResult('');
        })}, [])

  const {
    acceptedFiles,
    getRootProps,
    getInputProps,
    isDragActive,
    isDragAccept,
    isDragReject
  } = useDropzone({
    accept: 'image/jpeg, image/png',
    maxFiles: 1,
    onDrop
  });

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

  const archivos = acceptedFiles.map(file => (
    <li key={file.path}>
      {file.path} - {file.size} bytes
    </li>
  ));

  const sendFile = () => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('eye', 'Left');
    fetch(`http://localhost:8000/analize/${file.name}`, {
        method: 'POST',
        body: formData,
        headers: {
            "Authorization": "TOKEN f2e94d1c083f1f84117b5c515b028a2dcfc09776"
        }
    }).then(response => response.json())
    .then(result => {
      setResult(result);
      setFlagResult(true);
      console.log('Success:', result);
    })
    .catch(error => console.log('Error:', error));
  }

  const deleteImage = () => {
    const output = document.getElementById('image_loaded');
    // output.src = "";
    // const resultAlert = document.getElementById('result-alert');
    // resultAlert = '';
    // setResult(null);
    setFlagResult(false);
    setFlagFileLoaded(false);
  }

  const analysisResult = result.response === 'La imagen no es apta para ser procesada.' ?
  <Alert severity="error">
    {result.response}
  </Alert>
  :
  <Alert severity="success">
    {result.response}
  </Alert>

  return (
    <div className="container">
      <div {...getRootProps({style})}>
        <input {...getInputProps()} />
        <p>Arrastra y suelta la retinografía en esta región.</p>
      </div>
      {/* <aside style={thumbsContainer}>
        {thumbs}
      </aside> */}
      <br/>
        <Button onClick={sendFile} variant="outlined">Enviar</Button>
        <Button onClick={deleteImage} variant="outlined" color="error">Borrar</Button>
        {flagFileLoaded === true && <div>
          <h4>Vista previa:</h4>
          <img width="400px" id="image_loaded"/>
          <aside>
              <h4>Detalle:</h4>
              <ul>{archivos}</ul>
          </aside>
        </div>
        }
        {flagResult === true && 
          <div>
            <h4>Resultados:</h4>
            <div id="result-alert">
              {flagResult === false ? null : analysisResult}
            </div>
          </div>
        }
        <br/>
    </div>
  );
}

export default Basic;