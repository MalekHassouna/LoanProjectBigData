import { Component } from '@angular/core';
import { PredictionService } from 'src/app/services/prediction.service';
import axios from 'axios';
@Component({
  selector: 'app-prediction',
  templateUrl: './prediction.component.html',
  styleUrls: ['./prediction.component.css']
})
export class PredictionComponent {
  selectedFile: File | null = null;
  selectedModel: string = '';
  loading: boolean = false;

  constructor(private predictionService: PredictionService) { }

  onFileSelected(event: any): void {
    this.selectedFile = event.target.files[0];
  }

  onModelSelected(event: any): void {
    this.selectedModel = event.target.value;
  }

  onSubmit(): void {
    if (!this.selectedFile || !this.selectedModel) {
      alert('Please select both a file and a model!');
      return;
    }
    this.loading = true;
    this.predictionService.uploadFile(this.selectedFile, this.selectedModel).subscribe(
      (response: Blob) => {
        // Crée un lien pour télécharger le fichier
        const url = window.URL.createObjectURL(response);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'output_predictions.csv';
        a.click();
        this.loading = false;
      },
      (error) => {
        console.error('Error:', error);
        alert('An error occurred while processing your request.');
        this.loading = false;
      }
    );
  }

}
