import React from 'react';
import Avatar from '@material-ui/core/Avatar';
import Button from '@material-ui/core/Button';
import CssBaseline from '@material-ui/core/CssBaseline';
import TextField from '@material-ui/core/TextField';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import Checkbox from '@material-ui/core/Checkbox';
import Link from '@material-ui/core/Link';
import Paper from '@material-ui/core/Paper';
import Box from '@material-ui/core/Box';
import Grid from '@material-ui/core/Grid';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { makeStyles } from '@material-ui/core/styles';
import { useState } from 'react';
import axios from 'axios';
import { Redirect } from "react-router-dom";
import Alert from '@mui/material/Alert';
// import { signIn } from './signIn';

function Copyright() {
    return (
        <Typography variant="body2" color="textSecondary" align="center">
            {'Copyright © '}
            <Link color="inherit" href="https://material-ui.com/">
                Gecko - Analizador de Retinografías Online
            </Link>{' '}
            {new Date().getFullYear()}
            {'.'}
        </Typography>
    );
}

const useStyles = makeStyles((theme) => ({
    root: {
        height: '100vh',
    },
    image: {
        backgroundImage: 'url(https://www.icare-world.com/wp-content/uploads/2020/07/DRS_2-iCare.jpg)',
        backgroundRepeat: 'no-repeat',
        backgroundColor:
            theme.palette.type === 'light' ? theme.palette.grey[50] : theme.palette.grey[900],
        backgroundSize: 'cover',
        backgroundPosition: 'center',
    },
    paper: {
        margin: theme.spacing(8, 4),
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
    },
    avatar: {
        margin: theme.spacing(1),
        backgroundColor: theme.palette.secondary.main,
    },
    form: {
        width: '100%', // Fix IE 11 issue.
        marginTop: theme.spacing(1),
    },
    submit: {
        margin: theme.spacing(3, 0, 2),
    },
}));

const SignInSide = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const classes = useStyles();
    const [redirect, setRedirect] = useState('');
    const [alertPasswordIncorrect, setAlertPasswordIncorrect] = useState(false);

    // Lógica para el signIn
    const signIn = async (email, password) => {
        await axios.post('http://127.0.0.1:8000/login', 
            {
                'username': email,
                'password': password
            },
            {
                'headers': {
                    'Content-Type': 'application/json'
                }
            }
        ).then(response => {
            localStorage.setItem('user-token', JSON.stringify(response.data.token));
            setRedirect('/home');
        }).catch(error => setAlertPasswordIncorrect(true));
        // if (response.status === 200) {
        //     localStorage.setItem('user-token', JSON.stringify(response.data.token));
        //     setRedirect('/home');
        // } else if (response.status = 400) {
        //     setAlertPasswordIncorrect(true);
        // }
        // console.log("TOKEN: " + response.data.token);
        // console.log("response status code: " + response.status);
        // console.log(response.error)
        // if (response.error) console.log("Error: el email ingresado no se encuentra disponible");
        // else console.log("Logueo exitoso.");
    }

    const formRedirect = redirect ? <Redirect to={redirect} /> :
        <Grid container component="main" className={classes.root}>
            <CssBaseline />
            <Grid item xs={false} sm={4} md={7} className={classes.image} />
            <Grid item xs={12} sm={8} md={5} component={Paper} elevation={6} square>
                <div className={classes.paper}>
                    <Avatar className={classes.avatar}>
                        <LockOutlinedIcon />
                    </Avatar>
                    <Typography component="h1" variant="h5">
                        Sign In
                    </Typography>
                    <form className={classes.form} noValidate>
                        <TextField onChange={(event) => setEmail(event.target.value)}
                            variant="outlined"
                            margin="normal"
                            required
                            fullWidth
                            id="email"
                            label="Email"
                            name="email"
                            autoComplete="email"
                            placeholder="email@example.com"
                            value={email}
                            autoFocus
                        />
                        <TextField onChange={(event) => setPassword(event.target.value)}
                            variant="outlined"
                            margin="normal"
                            required
                            fullWidth
                            name="password"
                            label="Password"
                            type="password"
                            id="password"
                        />
                        {alertPasswordIncorrect === true && 
                            <Alert severity="error">El password igresado es incorrecto. Revisálo y volvé a intentar</Alert>
                        }
                        <FormControlLabel
                            control={<Checkbox value="remember" color="primary" />}
                            label="Remember me"
                        />
                        <Button onClick={(event) => {
                                event.preventDefault(); // Para evitar que recargue la página 
                                signIn(email, password);
                            }}
                            type="submit"
                            fullWidth
                            variant="contained"
                            color="primary"
                            className={classes.submit}
                        >
                            Sign In
                        </Button>
                        <Grid container>
                            <Grid item xs>
                                <Link href="#" variant="body2">
                                    Forgot password?
                                </Link>
                            </Grid>
                            <Grid item>
                                <Link href="#" variant="body2">
                                    {"Don't have an account? Sign Up"}
                                </Link>
                            </Grid>
                        </Grid>
                        <Box mt={5}>
                            <Copyright />
                        </Box>
                    </form>
                </div>
            </Grid>
        </Grid>;

    return (
        <div>
            {formRedirect}
        </div>
    )
}

export default SignInSide;