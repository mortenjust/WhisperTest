//
//  ContentView.swift
//  WhisperTest
//
//  Created by Morten Just on 11/24/22.
//

import SwiftUI
import AVFoundation
import whisper

struct ContentView: View {
    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            Text("Hello, world!")
        }
        .padding()
        .onAppear {
            doit()
        }
    }
    
    func floatArray() -> [Float] {
        
        let sample = "/Users/mortenjust/code/delete/whisper.cpp/samples/jfk.wav"
        let url = URL(fileURLWithPath: sample)
      
        let audioFile = try! AVAudioFile(forReading: url)
        let audioFormat = audioFile.processingFormat
        let audioFrameCount = UInt32(audioFile.length)
        let audioFileBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: audioFrameCount)

        try! audioFile.read(into: audioFileBuffer!)

        let array = audioFileBuffer!.array()
        
        return array
    }
    
    
    
    
    func doit() {
        /// https://github.com/ggerganov/whisper.cpp/blob/master/examples/main/main.cpp
        /// output to file: https://github.com/ggerganov/whisper.cpp/blob/master/examples/main/main.cpp#L244
        /// PCMBufferToFloatArray https://developer.apple.com/forums/thread/65772
        /// ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
        /// 16-bit WAV files
        ///
        
        let ctx = whisper_init("/Users/mortenjust/code/delete/whisper.cpp/models/ggml-base.en.bin")

        var params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)

        params.print_realtime       = true
        params.print_progress       = false
        params.print_timestamps     = true
        params.print_special_tokens = false
        params.translate            = false
        //params.language             = "en";
        params.n_threads            = 4
        params.offset_ms            = 0

        let n_samples = Int32(WHISPER_SAMPLE_RATE)
        let pcmf32 = [Float](repeating: 0, count: Int(n_samples))
        
        let floats = floatArray()
        

//        let ret = whisper_full(ctx, params, pcmf32, n_samples)
        let ret = whisper_full(ctx, params, floats, Int32(floats.count))
        assert(ret == 0, "Failed to run the model")

        let n_segments = whisper_full_n_segments(ctx)
        
        print("-- segments", n_segments)

        for i in 0..<n_segments {
            let text_cur = whisper_full_get_segment_text(ctx, i)
            print("-", text_cur as Any)
            
        }
        
        
        whisper_print_timings(ctx)
        whisper_free(ctx)

    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}



import AVFoundation

extension AudioBuffer {
    func array() -> [Float] {
        return Array(UnsafeBufferPointer(self))
    }
}

extension AVAudioPCMBuffer {
    func array() -> [Float] {
        return self.audioBufferList.pointee.mBuffers.array()
    }
}

extension Array where Element: FloatingPoint {
    mutating func buffer() -> AudioBuffer {
        return AudioBuffer(mNumberChannels: 1, mDataByteSize: UInt32(self.count * MemoryLayout<Element>.size), mData: &self)
    }
}
