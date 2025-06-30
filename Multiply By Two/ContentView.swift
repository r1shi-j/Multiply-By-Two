//
//  ContentView.swift
//  Multiply By Two
//
//  Created by Rishi Jansari on 30/06/2025.
//

import CoreML
import SwiftUI

struct ContentView: View {
    @State private var inputNumber = 0
    @State private var outputNumber = 0
    
    var body: some View {
        VStack {
            TextField("Enter number", value: $inputNumber, formatter: NumberFormatter())
            Text(String(outputNumber))
            Button("Multiply by Two!") {
                outputNumber = multiplyByTwo(inputNumber)
            }
        }
    }
    
    func multiplyByTwo(_ number: Int) -> Int {
        guard let model = try? MultiplyByTwo() else {
            fatalError("Could not create model")
        }
        guard let tensorInput = try? MLMultiArray(shape: [1, 1], dataType: .float32) else {
            fatalError("Could not create tensorInput")
        }
        tensorInput[0] = NSNumber(value: Float32(number))
        do {
            let prediction = try model.prediction(dense_input: tensorInput)
            return prediction.Identity[0].intValue
        } catch {
            fatalError("Could not create prediction")
        }
    }
}

#Preview {
    ContentView()
}
