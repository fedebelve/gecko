import axios from 'axios';

const signIn = async (email, password) => {
    const response = await axios.post('http://127.0.0.1:8000/login', 
        {
            username: email,
            password: password
        },
        {
            'headers': {
                'Content-Type': 'application/json'
            }
        }
    );
    localStorage.setItem('userToken', response.data.token)
}

export { signIn };