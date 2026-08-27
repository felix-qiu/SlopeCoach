import SwiftUI

struct PrimaryButton: View {
    let title: String
    var systemImage: String? = nil
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                if let systemImage {
                    Image(systemName: systemImage)
                        .font(.system(size: 17, weight: .semibold))
                }

                Text(title)
                    .font(.system(size: 17, weight: .semibold))
            }
                .frame(maxWidth: .infinity)
                .frame(height: 52)
        }
        .buttonStyle(.plain)
        .foregroundStyle(.white)
        .background(Color.slopePrimary, in: RoundedRectangle(cornerRadius: 15, style: .continuous))
        .shadow(color: Color.slopePrimary.opacity(0.24), radius: 12, y: 6)
        .accessibilityIdentifier("primary-button-\(title)")
    }
}
