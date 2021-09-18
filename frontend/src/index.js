import React from 'react';
import ReactDOM from 'react-dom';
// import './index.css';
import App from './components/App/App';
import SignInSide from './components/SignIn/SignIn';
import reportWebVitals from './reportWebVitals';
import ImageLoadScreen from './components/ImageLoadScreen/ImageLoadScreen';
import ImageUploadButton from "./components/ImageLoadScreen/ImageUploadButton/ImageUploadButton";
import FileList from './components/ImageLoadScreen/DragAndDrop/FileList'
import FileUpload from './components/ImageLoadScreen/FileUpload/FileUpload';
import Basic from './components/ImageLoadScreen/DropZone/Basic';
import Previews from './components/ImageLoadScreen/DropZone/Previews';
import ImageUpload from './components/ImageLoadScreen/ImageUpload/ImageUpload';

ReactDOM.render(
  <React.StrictMode>
    {/* <Basic /> */}
    {/* <Previews/> */}
    <ImageUpload/>
  </React.StrictMode>,
  document.getElementById('root')
);

// If you want to start measuring performance in your app, pass a function
// to log results (for example: reportWebVitals(console.log))
// or send to an analytics endpoint. Learn more: https://bit.ly/CRA-vitals
reportWebVitals();