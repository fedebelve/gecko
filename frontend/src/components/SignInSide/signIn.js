import axios from 'axios';

const signIn = async (email, password) => {
    const response = await axios.post('http://127.0.0.1:8000/login', 
        {
            'username': email,
            'password': password
        },
        {
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    );
    console.log("response status code: " + response.status);
    console.log(response.error)
    if (response.error) console.log("Error: el email ingresado no se encuentra disponible");
    else console.log("Logueo exitoso.");
}

export default signIn;