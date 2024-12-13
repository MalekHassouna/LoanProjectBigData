import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PredictionService {
  private apiUrl = 'http://localhost:5000/predict';

  constructor(private http: HttpClient) {}

  uploadFile(file: File, model: string): Observable<Blob> {
    const formData: FormData = new FormData();
    formData.append('file', file, file.name);
    formData.append('model', model);

    return this.http.post(this.apiUrl, formData, { responseType: 'blob' });
  }
}
