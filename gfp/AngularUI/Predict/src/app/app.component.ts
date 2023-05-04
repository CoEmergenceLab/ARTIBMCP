import { Component, OnInit } from '@angular/core';
import { FormBuilder, FormGroup, Validators } from '@angular/forms';
import { MatSnackBar } from '@angular/material/snack-bar';
import { MatDialog, MatDialogRef } from '@angular/material/dialog';
import { DomSanitizer } from '@angular/platform-browser';
import { HttpClient } from '@angular/common/http';


interface ClusterInfo{
  cluster:number
}

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  name: string | null = null;
  description: string | null = null;
  selectedFile: File | null = null;
  imageSrc: string | null = null;
  imageName: string | null = null;
  uploading: boolean = false;
  progress: number = 0;

  clusterId:number = -1;

  constructor(private http:HttpClient){}

  onFileSelected(event:Event) {
    this.clusterId = -1;
    const input = event.target as HTMLInputElement;
    const file = input.files ? input.files[0] : null;
    if (file) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = () => {
        this.imageSrc = reader.result as string;
        // Set the src attribute of the <img> element to the data URL
        const img = document.getElementById('preview') as HTMLImageElement;
        img.src = this.imageSrc;
      };
      reader.readAsDataURL(this.selectedFile);
      this.imageName = this.selectedFile.name;
    } else {
      this.imageSrc = null;
      this.imageName = null;
    }
  }

  onUpload() {
    if (!this.selectedFile) {
      return;
    }

    this.uploading = true;
    const reader = new FileReader();
    reader.readAsDataURL(this.selectedFile);
    reader.onload = () => {
      const imageData = reader.result?.toString().split(',')[1];
      const payload = {
        image: imageData
      };
      this.http.post('http://127.0.0.1:8002/predict', payload).subscribe(
        (response:any) => {
          console.log(response);
          if(response){
            this.clusterId = response.cluster
          }

        },
        (error) => {
          console.log(error);
        }
      );
    };
  

  
    // Simulate the upload progress for demonstration purposes
    // const interval = setInterval(() => {
    //   this.progress += 5;
    //   if (this.progress === 100) {
    //     clearInterval(interval);
    //     this.uploading = false;
    //     this.progress = 0;
    //     this.selectedFile = null;
    //     this.name = null;
    //     this.description = null;
    //     this.imageSrc = null;
    //     this.imageName = null;
    //   }
    // }, 500);
  }
}