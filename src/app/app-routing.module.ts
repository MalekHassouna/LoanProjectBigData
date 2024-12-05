import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { HomeComponent } from './components/home/home.component';
import { LoginComponent } from './components/login/login.component';
import { SignupComponent } from './components/signup/signup.component';
import { PredictionComponent } from './components/prediction/prediction.component';
import { TeamComponent } from './components/team/team.component';

const routes: Routes = [
  {path:"" , component: HomeComponent},
  //http://localhost:4200/connexion => app-login will be displayed 
  {path:"connexion", component: LoginComponent },
  //http://localhost:4200/insc => app-signup will be displayed 
  {path:"insc" , component: SignupComponent},
  {path:"prediction" , component: PredictionComponent},
  {path:"team" , component: TeamComponent},
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule {
  
 }
