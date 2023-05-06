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
  selectedFiles:File[] =[];
  clusterId:number = -1;
  distanceFromCluster = 0;
  averageDistance = 0;
  dataUrls: string[] = [];
  constructor(private http:HttpClient){}

  onFileSelected(event:Event) {
    this.clusterId = -1;
    const input = event.target as HTMLInputElement;
    const file = input.files ? input.files[0] : null;
    this.selectedFiles = [];
    this.dataUrls =[];
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

  onNewFileSelected(event:Event) {
    this.clusterId = -1;
    this.imageSrc = null;
    this.imageName = null;
    const input = event.target as HTMLInputElement;
    const files = input.files ? Array.from(input.files) : null;
    if (files) {
      this.selectedFiles = files;

      for (const file of this.selectedFiles) {
        const reader = new FileReader();
        reader.onload = () => {
         this.dataUrls.push(reader.result as string);
          if (this.dataUrls.length === this.selectedFiles.length) {
            // All files have been loaded, update the <img> elements to display them
            const imgs = document.querySelectorAll('.preview') as NodeListOf<HTMLImageElement>;
            for (let i = 0; i < imgs.length; i++) {
              imgs[i].src = this.dataUrls[i];
            }
          }
        };
        reader.readAsDataURL(file);
      }
    } else {
      this.selectedFiles = [];
      this.dataUrls =[];
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
            this.clusterId = response.cluster;
            this.distanceFromCluster = response.img_dist;
            this.averageDistance = response.threshold;
          }

        },
        (error) => {
          console.log(error);
        }
      );
    };
  }
  onAllUpload() {
    if (!this.selectedFiles || this.selectedFiles.length === 0) {
      return;
    }
  
    this.uploading = true;
    const promises = [];
    for (const file of this.selectedFiles) {
      const promise = new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onload = () => {
          const imageData = reader.result?.toString().split(',')[1];
          resolve(imageData);
        };
        reader.onerror = (error) => {
          reject(error);
        };
      });
      promises.push(promise);
    }
  
    Promise.all(promises).then((imageDataArray) => {
      const payload = {
        images: imageDataArray
      };
      this.http.post('http://127.0.0.1:8002/retrain', payload).subscribe(
        (response: any) => {
          console.log(response);
          if (response) {
            // this.clusterId = response.cluster;
            // this.distanceFromCluster = response.img_dist;
            // this.averageDistance = response.threshold;
          }
        },
        (error) => {
          console.log(error);
        }
      );
    });
  }
}